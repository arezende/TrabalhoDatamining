import geopandas as gpd
import pandas as pd
import numpy as np
import psycopg2
from shapely.geometry import Point, LineString, MultiLineString
from shapely.ops import linemerge, unary_union
from sklearn.cluster import DBSCAN
import folium
import warnings

# --- CONFIGURAções GLOBAIS ---
warnings.filterwarnings('ignore', 'UserWarning', module='geopandas')
pd.options.mode.chained_assignment = None

DB_CONFIG = {
    'user': 'postgres',
    'password': 'P@ssw0rd123',
    'host': '172.18.166.112',
    'port': '5432',
    'dbname': 'onibus_db'
}

# --- PARÂMETROS AJUSTÁVEIS ---
PRIMARY_TERMINAL_EPS = 200
PRIMARY_TERMINAL_MIN_SAMPLES = 20 # Reduzido um pouco para mais flexibilidade
FALLBACK_TERMINAL_EPS = 0.0025
FALLBACK_TERMINAL_MIN_SAMPLES = 5
OUTLIER_EPS_METERS = 15
OUTLIER_MIN_SAMPLES = 10

class RouteProcessor:
    def __init__(self, db_params):
        self.db_params = db_params
        self.conn = None
        self.terminal_A = None
        self.terminal_B = None
        self.garage_point_ids = []

    def connect(self):
        try:
            self.conn = psycopg2.connect(**self.db_params, client_encoding='UTF8')
            print("✅ Conexão estabelecida.")
        except Exception as e:
            print(f"❌ Erro ao conectar: {e}")

    def close(self):
        if self.conn:
            self.conn.close()
            print("\n🔌 Conexão fechada.")

    def setup_database(self):
        print("🔧 Verificando e preparando o banco de dados...")
        with self.conn.cursor() as cursor:
            cursor.execute("ALTER TABLE pontos_gps_treino ADD COLUMN IF NOT EXISTS is_outlier BOOLEAN;")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS rotas_referencia (
                    id SERIAL PRIMARY KEY, linha VARCHAR(20) NOT NULL, sentido VARCHAR(10) NOT NULL,
                    geom GEOMETRY(LineString, 4326), UNIQUE(linha, sentido)
                );
            """)
            self.conn.commit()
        print("  ✔️ Estrutura do DB está pronta.")

    def _fetch_data(self, linha_id, only_inliers=False):
        where_clauses = ["linha = %s"]
        params = [linha_id]
        if only_inliers:
            where_clauses.append("is_outlier IS NOT TRUE")
        sql = f"""
            SELECT id, ordem, datahora_servidor, velocidade, geom FROM pontos_gps_treino
            WHERE {' AND '.join(where_clauses)} ORDER BY ordem, datahora_servidor;
        """
        gdf = gpd.read_postgis(sql, self.conn, params=params, geom_col='geom')
        if gdf.empty:
            print(f"  ⚠️ Nenhum dado encontrado para os filtros atuais."); return None
        return gdf

    def _delete_existing_route(self, linha_id):
        sql = "DELETE FROM rotas_referencia WHERE linha = %s;"
        with self.conn.cursor() as cursor:
            cursor.execute(sql, (linha_id,))
            self.conn.commit()

    def _save_route_to_db(self, linha_id, sentido, route_geom):
        if route_geom is None or route_geom.is_empty:
            print(f"    ⚠️ Rota '{sentido}' vazia, não foi salva.")
            return
        sql = "INSERT INTO rotas_referencia (linha, sentido, geom) VALUES (%s, %s, ST_GeomFromText(%s, 4326));"
        with self.conn.cursor() as cursor:
            cursor.execute(sql, (linha_id, sentido, route_geom.wkt))
            self.conn.commit()
        print(f"    ✔️ Rota '{sentido}' salva no banco de dados.")

    def classify_outliers(self, linha_id):
        print("\n--- ETAPA 1: Classificando Outliers ---")
        gdf = self._fetch_data(linha_id)
        if gdf is None: return False
        
        print("  - Rodando DBSCAN para encontrar o cluster principal...")
        coords_rad = np.radians(gdf['geom'].apply(lambda p: (p.y, p.x)).tolist())
        eps_rad = OUTLIER_EPS_METERS / 6371000
        db = DBSCAN(eps=eps_rad, min_samples=OUTLIER_MIN_SAMPLES, metric='haversine').fit(coords_rad)
        gdf['cluster'] = db.labels_

        if len(gdf['cluster'].unique()) > 1 and -1 in gdf['cluster'].values:
            main_cluster_label = gdf[gdf['cluster'] != -1]['cluster'].value_counts().idxmax()
            gdf['is_outlier'] = (gdf['cluster'] != main_cluster_label)
        else:
            gdf['is_outlier'] = False

        print(f"  - {gdf['is_outlier'].sum()} outliers identificados.")
        self._plot_outlier_map(gdf, linha_id)
        
        print("  - Atualizando o banco de dados...")
        with self.conn.cursor() as cursor:
            cursor.execute("UPDATE pontos_gps_treino SET is_outlier = NULL WHERE linha = %s", (linha_id,))
            outlier_ids = tuple(gdf[gdf['is_outlier']].id.tolist())
            if outlier_ids: cursor.execute("UPDATE pontos_gps_treino SET is_outlier = TRUE WHERE id IN %s", (outlier_ids,))
            inlier_ids = tuple(gdf[~gdf['is_outlier']].id.tolist())
            if inlier_ids: cursor.execute("UPDATE pontos_gps_treino SET is_outlier = FALSE WHERE id IN %s", (inlier_ids,))
            self.conn.commit()
        return True

    # <<< MÉTODO REESCRITO PARA DISTINGUIR GARAGEM E TERMINAL >>>
    def _find_terminals(self, gdf):
        print("--- ETAPA 2: Detectando Terminais e Garagens ---")
        
        # 1. Identificar paradas e garagens por tempo
        terminal_candidate_points = []
        self.garage_point_ids = []
        gdf_proj = gdf.to_crs(31983)
        
        for ordem, group in gdf_proj.groupby('ordem'):
            group = group.sort_values('datahora_servidor')
            group['time_diff'] = group['datahora_servidor'].diff().dt.total_seconds()
            group['dist_diff'] = group.geometry.distance(group.geometry.shift())
            group['is_stopped'] = (group['dist_diff'] < 15) & (group['time_diff'] < 180) # Parada de até 3min
            group['stop_block'] = (group['is_stopped'] != group['is_stopped'].shift()).cumsum()
            
            stopped_blocks = group[group['is_stopped']].groupby('stop_block')
            for _, stop_block in stopped_blocks:
                duration = stop_block['time_diff'].sum()
                if 300 <= duration <= 2400: # Parada de 5 a 40 minutos (TERMINAL)
                    terminal_candidate_points.append(stop_block)
                elif duration > 2400: # Parada > 40 minutos (GARAGEM)
                    self.garage_point_ids.extend(stop_block['id'].tolist())

        print(f"  - Identificados {len(self.garage_point_ids)} pontos de garagem (serão ignorados).")
        
        # 2. Método Primário: Clusterizar os pontos candidatos a terminal
        if terminal_candidate_points:
            gdf_stops = gpd.GeoDataFrame(pd.concat(terminal_candidate_points), geometry='geom', crs=31983)
            print(f"  - {len(gdf_stops)} pontos candidatos a terminal encontrados.")
            if len(gdf_stops) > PRIMARY_TERMINAL_MIN_SAMPLES * 2:
                coords = np.array([[p.x, p.y] for p in gdf_stops.geometry])
                db = DBSCAN(eps=PRIMARY_TERMINAL_EPS, min_samples=PRIMARY_TERMINAL_MIN_SAMPLES).fit(coords)
                gdf_stops['cluster'] = db.labels_
                cluster_counts = gdf_stops[gdf_stops['cluster'] != -1]['cluster'].value_counts()
                if len(cluster_counts) >= 2:
                    top_clusters = cluster_counts.head(2).index.tolist()
                    t1 = unary_union(gdf_stops[gdf_stops['cluster'] == top_clusters[0]].geometry).centroid
                    t2 = unary_union(gdf_stops[gdf_stops['cluster'] == top_clusters[1]].geometry).centroid
                    terminals_wgs84 = gpd.GeoSeries([t1, t2], crs=31983).to_crs(4326)
                    self.terminal_A, self.terminal_B = terminals_wgs84.iloc[0], terminals_wgs84.iloc[1]
                    if self.terminal_A.x > self.terminal_B.x: self.terminal_A, self.terminal_B = self.terminal_B, self.terminal_A
                    print("  ✔️ Terminais encontrados com o método primário (paradas longas).")
                    return True

        # 3. Método de Fallback: Se o primário falhar
        print("  ⚠️ Método primário falhou. Tentando fallback (pontos de extremidade).")
        gdf_no_garage = gdf[~gdf['id'].isin(self.garage_point_ids)] # Ignora garagens
        endpoints = gdf_no_garage.groupby('ordem').agg(first_geom=('geom', 'first'), last_geom=('geom', 'last'))
        all_endpoints = gpd.GeoDataFrame(geometry=pd.concat([endpoints.first_geom, endpoints.last_geom], ignore_index=True), crs=gdf.crs)
        
        if len(all_endpoints) < FALLBACK_TERMINAL_MIN_SAMPLES * 2:
            print("  ❌ Pontos de extremidade insuficientes."); return False

        coords = np.array([[p.x, p.y] for p in all_endpoints.geometry])
        db = DBSCAN(eps=FALLBACK_TERMINAL_EPS, min_samples=FALLBACK_TERMINAL_MIN_SAMPLES).fit(coords)
        all_endpoints['cluster'] = db.labels_
        cluster_counts = all_endpoints[all_endpoints['cluster'] != -1]['cluster'].value_counts()
        
        if len(cluster_counts) < 2:
            print("  ❌ Método de fallback também falhou."); return False

        top_clusters = cluster_counts.head(2).index.tolist()
        self.terminal_A = unary_union(all_endpoints[all_endpoints['cluster'] == top_clusters[0]].geometry).centroid
        self.terminal_B = unary_union(all_endpoints[all_endpoints['cluster'] == top_clusters[1]].geometry).centroid
        if self.terminal_A.x > self.terminal_B.x: self.terminal_A, self.terminal_B = self.terminal_B, self.terminal_A
        print("  ✔️ Terminais encontrados com o método de fallback.")
        return True

    def generate_and_save_routes(self, linha_id):
        gdf = self._fetch_data(linha_id, only_inliers=True)
        if gdf is None: return

        if not self._find_terminals(gdf):
            print(f"  ❌ Processamento da linha {linha_id} cancelado: não foi possível identificar 2 terminais."); return

        # Remover pontos de garagem antes de segmentar
        gdf = gdf[~gdf['id'].isin(self.garage_point_ids)]

        print("--- ETAPA 3: Segmentando Viagens e Gerando Rotas ---")
        ida_trips, volta_trips = [], []
        tolerance = 0.007
        for ordem, vehicle_df in gdf.groupby('ordem'):
            if len(vehicle_df) < 5: continue
            vehicle_df['dist_A'] = vehicle_df.geometry.distance(self.terminal_A)
            vehicle_df['dist_B'] = vehicle_df.geometry.distance(self.terminal_B)
            vehicle_df['location_state'] = 0
            vehicle_df.loc[vehicle_df['dist_A'] < tolerance, 'location_state'] = 1
            vehicle_df.loc[vehicle_df['dist_B'] < tolerance, 'location_state'] = 2
            vehicle_df['block_id'] = (vehicle_df['location_state'] != vehicle_df['location_state'].shift()).cumsum()
            blocks = [{'state': b.iloc[0]['location_state'], 'df': b} for _, b in vehicle_df.groupby('block_id') if len(b) >= 2]
            for i in range(len(blocks) - 2):
                if blocks[i]['state'] == 1 and blocks[i+1]['state'] == 0 and blocks[i+2]['state'] == 2:
                    ida_trips.append(blocks[i+1]['df'])
                elif blocks[i]['state'] == 2 and blocks[i+1]['state'] == 0 and blocks[i+2]['state'] == 1:
                    volta_trips.append(blocks[i+1]['df'])
        
        print(f"  - Identificadas {len(ida_trips)} viagens de Ida e {len(volta_trips)} viagens de Volta.")
        self._plot_direction_points_map(ida_trips, volta_trips, linha_id)
        
        print("  - Gerando rotas finais a partir dos pontos...")
        route_ida = self._create_route_from_trips(ida_trips)
        route_volta = self._create_route_from_trips(volta_trips)
        
        print("  - Salvando rotas no banco de dados...")
        self._delete_existing_route(linha_id)
        self._save_route_to_db(linha_id, 'Ida', route_ida)
        self._save_route_to_db(linha_id, 'Volta', route_volta)
        
        self._plot_final_routes_map(ida_trips, volta_trips, route_ida, route_volta, linha_id)

    def _create_route_from_trips(self, trips):
        if not trips: return None
        trip_lines = [LineString(t.sort_values('datahora_servidor').geometry.tolist()) for t in trips if len(t) > 1]
        if not trip_lines: return None
        merged = linemerge(trip_lines)
        return max(merged.geoms, key=lambda line: line.length) if isinstance(merged, MultiLineString) else merged
    
    def _plot_outlier_map(self, gdf, linha_id):
        print("  - Gerando mapa de validação de outliers...")
        m = folium.Map(location=[gdf.unary_union.centroid.y, gdf.unary_union.centroid.x], zoom_start=12)
        inliers = gdf[~gdf['is_outlier']]
        outliers = gdf[gdf['is_outlier']]
        inlier_group = folium.FeatureGroup(name=f"Inliers ({len(inliers)})").add_to(m)
        for p in inliers.iloc[::20].geometry: folium.CircleMarker((p.y, p.x), radius=1, color='blue', fill_opacity=0.7).add_to(inlier_group)
        outlier_group = folium.FeatureGroup(name=f"Outliers ({len(outliers)})").add_to(m)
        for p in outliers.geometry: folium.CircleMarker((p.y, p.x), radius=2, color='red', fill_opacity=0.9).add_to(outlier_group)
        folium.LayerControl().add_to(m)
        m.save(f'mapa_1_outliers_{linha_id}.html')
        print(f"  ✔️ Mapa de outliers salvo.")

    def _plot_direction_points_map(self, ida_trips, volta_trips, linha_id):
        print("  - Gerando mapa de validação de sentidos...")
        center_point = self.terminal_A if self.terminal_A else None
        if not center_point and (ida_trips or volta_trips):
            all_trips = pd.concat(ida_trips + volta_trips)
            center_point = all_trips.unary_union.centroid
        if not center_point: print("   - Não foi possível centralizar o mapa de sentidos."); return

        m = folium.Map(location=[center_point.y, center_point.x], zoom_start=12)
        if ida_trips:
            ida_group = folium.FeatureGroup(name="Pontos Ida").add_to(m)
            for p in pd.concat(ida_trips).iloc[::20].geometry: folium.CircleMarker((p.y, p.x), radius=1, color='blue').add_to(ida_group)
        if volta_trips:
            volta_group = folium.FeatureGroup(name="Pontos Volta").add_to(m)
            for p in pd.concat(volta_trips).iloc[::20].geometry: folium.CircleMarker((p.y, p.x), radius=1, color='red').add_to(volta_group)
        if self.terminal_A: folium.Marker([self.terminal_A.y, self.terminal_A.x], popup="Terminal A", icon=folium.Icon(color='green', icon='play')).add_to(m)
        if self.terminal_B: folium.Marker([self.terminal_B.y, self.terminal_B.x], popup="Terminal B", icon=folium.Icon(color='purple', icon='stop')).add_to(m)
        folium.LayerControl().add_to(m)
        m.save(f'mapa_2_sentidos_{linha_id}.html')
        print(f"  ✔️ Mapa de pontos de ida/volta salvo.")

    def _plot_final_routes_map(self, ida_trips, volta_trips, route_ida, route_volta, linha_id):
        print("  - Gerando mapa de validação final...")
        center_point = route_ida.centroid if route_ida else (self.terminal_A if self.terminal_A else None)
        if not center_point and (ida_trips or volta_trips):
            all_trips = pd.concat(ida_trips + volta_trips)
            center_point = all_trips.unary_union.centroid
        if not center_point: print("   - Não foi possível gerar mapa final."); return
        
        m = folium.Map(location=[center_point.y, center_point.x], zoom_start=12)
        if ida_trips:
            ida_group = folium.FeatureGroup(name="Pontos Ida").add_to(m)
            for p in pd.concat(ida_trips).iloc[::20].geometry: folium.CircleMarker((p.y, p.x), radius=1, color='lightblue', fill_opacity=0.5).add_to(ida_group)
        if volta_trips:
            volta_group = folium.FeatureGroup(name="Pontos Volta").add_to(m)
            for p in pd.concat(volta_trips).iloc[::20].geometry: folium.CircleMarker((p.y, p.x), radius=1, color='lightcoral', fill_opacity=0.5).add_to(volta_group)
        if route_ida: folium.GeoJson(route_ida, name="Rota Ida Final", style_function=lambda x: {'color':'blue', 'weight':5}).add_to(m)
        if route_volta: folium.GeoJson(route_volta, name="Rota Volta Final", style_function=lambda x: {'color':'red', 'weight':5}).add_to(m)
        if self.terminal_A: folium.Marker([self.terminal_A.y, self.terminal_A.x], popup="Terminal A", icon=folium.Icon(color='green', icon='play')).add_to(m)
        if self.terminal_B: folium.Marker([self.terminal_B.y, self.terminal_B.x], popup="Terminal B", icon=folium.Icon(color='purple', icon='stop')).add_to(m)
        folium.LayerControl().add_to(m)
        m.save(f'mapa_3_final_{linha_id}.html')
        print(f"  ✔️ Mapa final salvo.")


if __name__ == '__main__':
    linhas_para_processar = [
        '483', '864', '639', '3', '309', '774', '629', '371', '397', '100', 
        '838', '315', '624', '388', '918', '665', '328', '497', '878', '355'
    ]
    processor = RouteProcessor(DB_CONFIG)
    processor.connect()
    if processor.conn:
        try:
            processor.setup_database()
            for linha in linhas_para_processar:
                print("\n" + "#"*60); print(f"### PROCESSANDO LINHA: {linha} ###"); print("#"*60)
                if processor.classify_outliers(linha):
                    processor.generate_and_save_routes(linha)
        finally:
            processor.close()
    print("\n" + "="*50); print("🎉 Processo de criação de rotas concluído!"); print("="*50)
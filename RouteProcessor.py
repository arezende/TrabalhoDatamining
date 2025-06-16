import geopandas as gpd
import pandas as pd
import numpy as np
import psycopg2
from shapely.geometry import Point, LineString, MultiPoint
from sklearn.cluster import DBSCAN, KMeans
import folium
import warnings

warnings.filterwarnings('ignore', 'UserWarning', module='geopandas')
pd.options.mode.chained_assignment = None

# --- CONFIGURAÇÕES - AJUSTE AQUI ---
DB_CONFIG = {
    'user': 'postgres',
    'password': 'P@ssw0rd123',
    'host': '172.18.166.112',
    'port': '5432',
    'dbname': 'onibus_db'
}

class RouteProcessor:
    def __init__(self, db_params):
        self.db_params = db_params
        self.conn = None

    def connect(self):
        try:
            self.conn = psycopg2.connect(**self.db_params, client_encoding='latin1')
            print("✅ Conexão estabelecida.")
        except Exception as e: print(f"❌ Erro ao conectar: {e}")

    def close(self):
        if self.conn: self.conn.close(); print("\n🔌 Conexão fechada.")

    def setup_routes_table(self):
        sql = "CREATE TABLE IF NOT EXISTS rotas_referencia (id SERIAL PRIMARY KEY, linha VARCHAR(20) NOT NULL, sentido VARCHAR(10) NOT NULL, geom GEOMETRY(LineString, 4326), UNIQUE(linha, sentido));"
        with self.conn.cursor() as cursor:
            cursor.execute(sql); self.conn.commit()
        print("🔧 Tabela 'rotas_referencia' pronta.")

    def _delete_existing_route(self, linha_id):
        print("  - Excluindo rotas antigas...")
        sql = "DELETE FROM rotas_referencia WHERE linha = %s;"
        with self.conn.cursor() as cursor:
            cursor.execute(sql, (linha_id,)); self.conn.commit()

    def _fetch_data(self, linha_id):
        print(f"1. Buscando dados da linha '{linha_id}'...")
        sql = f"SELECT id, ordem, datahora_servidor, velocidade, geom FROM pontos_gps_treino WHERE linha = '{linha_id}' AND EXTRACT(HOUR FROM datahora_servidor) BETWEEN 8 AND 22 ORDER BY ordem, datahora_servidor;"
        gdf = gpd.read_postgis(sql, self.conn, params=[linha_id], geom_col='geom')
        if gdf.empty:
            print(f"  ⚠️ Nenhum dado encontrado."); return None
        print(f"  ✔️ {len(gdf)} pontos carregados.")
        return gdf

    def _create_reference_route(self, gdf_direction, smoothing_window=35):
        if gdf_direction.empty or len(gdf_direction) < smoothing_window: return None
        print(f"  - Criando rota de referência com suavização...")
        gdf_proj = gdf_direction.to_crs(epsg=31983)
        multipoint = MultiPoint(gdf_proj.geometry.tolist())
        hull = multipoint.convex_hull
        if hull.is_empty or not hasattr(hull, 'exterior'): return None
        points = list(hull.exterior.coords)
        max_dist, p1, p2 = 0, None, None
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                dist = Point(points[i]).distance(Point(points[j]))
                if dist > max_dist: max_dist, p1, p2 = dist, Point(points[i]), Point(points[j])
        if not p1: return None
        main_axis = LineString([p1, p2])
        gdf_proj['dist_on_axis'] = gdf_proj.geometry.apply(lambda p: main_axis.project(p))
        sorted_points = gdf_proj.sort_values('dist_on_axis')
        coords = sorted_points.geometry.apply(lambda p: (p.x, p.y)).tolist()
        rolling_mean = pd.DataFrame(coords).rolling(window=smoothing_window, center=True, min_periods=1).mean().dropna()
        if rolling_mean.empty: return None
        smooth_route_proj = LineString(rolling_mean.values)
        return gpd.GeoSeries([smooth_route_proj], crs=31983).to_crs(4326).iloc[0]

    def _save_route_to_db(self, linha_id, sentido, route_geom):
        if route_geom is None: print(f"    ⚠️ Rota '{sentido}' vazia, não salva."); return
        sql = "INSERT INTO rotas_referencia (linha, sentido, geom) VALUES (%s, %s, ST_GeomFromText(%s, 4326));"
        with self.conn.cursor() as cursor:
            cursor.execute(sql, (linha_id, sentido, route_geom.wkt)); self.conn.commit()
        print(f"    ✔️ Rota '{sentido}' salva.")

    def _plot_validation_map(self, gdf_ida, gdf_volta, route_ida, route_volta, linha_id):
        print("4. Gerando mapa de validação...")
        center_gdf = gdf_ida if not gdf_ida.empty else gdf_volta
        if center_gdf.empty: return
        center_point = center_gdf.dissolve().centroid.iloc[0]
        m = folium.Map(location=[center_point.y, center_point.x], zoom_start=13)
        if not gdf_ida.empty:
            ida_group = folium.FeatureGroup(name=f"Pontos Ida ({len(gdf_ida)})").add_to(m)
            for _, r in gdf_ida.iloc[::20].iterrows(): folium.CircleMarker((r.geom.y, r.geom.x), radius=1.5, color='blue', fill_opacity=0.7).add_to(ida_group)
        if not gdf_volta.empty:
            volta_group = folium.FeatureGroup(name=f"Pontos Volta ({len(gdf_volta)})").add_to(m)
            for _, r in gdf_volta.iloc[::20].iterrows(): folium.CircleMarker((r.geom.y, r.geom.x), radius=1.5, color='red', fill_opacity=0.7).add_to(volta_group)
        if route_ida: folium.GeoJson(route_ida, name="Rota Ida (Gerada)", style_function=lambda x: {'color': 'blue', 'weight': 5, 'opacity': 0.8}).add_to(m)
        if route_volta: folium.GeoJson(route_volta, name="Rota Volta (Gerada)", style_function=lambda x: {'color': 'red', 'weight': 5, 'opacity': 0.8}).add_to(m)
        folium.LayerControl().add_to(m)
        m.save(f'mapa_validacao_final_{linha_id}.html'); print(f"  ✔️ Mapa salvo.")

    def processar_linha(self, linha_id):
        print("\n" + "="*50); print(f"🚀 Processando Linha: {linha_id}")
        
        # Etapa 1: Limpeza de Outliers
        gdf = self._fetch_data(linha_id)
        if gdf is None: return

        print("  - Removendo outliers com DBSCAN...")
        coords_rad = np.radians(gdf[['geom']].apply(lambda r: (r.geom.y, r.geom.x), axis=1).tolist())
        db = DBSCAN(eps=50/6371000, min_samples=20, metric='haversine').fit(coords_rad)
        gdf['cluster'] = db.labels_
        if -1 in gdf['cluster'].unique() and len(gdf['cluster'].value_counts()) > 1:
            main_cluster_label = gdf[gdf['cluster'] != -1]['cluster'].value_counts().idxmax()
            gdf_clean = gdf[gdf['cluster'] == main_cluster_label].copy()
        else:
            gdf_clean = gdf.copy()
        print(f"  - {len(gdf) - len(gdf_clean)} outliers removidos.")

        # Etapa 2: Identificação de Terminais
        print("2. Identificando terminais...")
        low_speed_gdf = gdf_clean[gdf_clean['velocidade'] <= 5].copy()
        if len(low_speed_gdf) < 50: low_speed_gdf = gdf_clean.copy()
        eps_degrees = 300 / 111320.0
        coords = np.array(low_speed_gdf['geom'].apply(lambda p: (p.x, p.y)).tolist())
        db_terminals = DBSCAN(eps=eps_degrees, min_samples=20).fit(coords)
        low_speed_gdf['terminal_cluster'] = db_terminals.labels_
        cluster_counts = low_speed_gdf[low_speed_gdf['terminal_cluster'] != -1]['terminal_cluster'].value_counts()
        if len(cluster_counts) < 2: print(f"  ❌ Não foi possível identificar 2 terminais distintos."); return
        
        top_clusters = cluster_counts.head(2).index.tolist()
        terminal_A = MultiPoint(low_speed_gdf[low_speed_gdf['terminal_cluster'] == top_clusters[0]].geometry.tolist()).centroid
        terminal_B = MultiPoint(low_speed_gdf[low_speed_gdf['terminal_cluster'] == top_clusters[1]].geometry.tolist()).centroid
        if terminal_A.x > terminal_B.x: terminal_A, terminal_B = terminal_B, terminal_A
        print("  - Terminais A e B definidos.")
        
        # Etapa 3: Classificação de Sentido baseada nas viagens entre terminais
        print("3. Classificando viagens por proximidade aos terminais...")
        gdf_clean['sentido'] = 'Indefinido'
        gdf_ida_list, gdf_volta_list = [], []

        for ordem, group in gdf_clean.groupby('ordem'):
            group['dist_a'] = group.geometry.distance(terminal_A)
            group['dist_b'] = group.geometry.distance(terminal_B)
            
            # Detecta quando o ônibus está "em um terminal"
            group['at_terminal_a'] = group['dist_a'] < (400 / 111320.0)
            group['at_terminal_b'] = group['dist_b'] < (400 / 111320.0)
            
            # Uma "mudança de estado" ocorre quando o status do terminal muda
            group['block'] = (group['at_terminal_a'] != group['at_terminal_a'].shift()) | \
                             (group['at_terminal_b'] != group['at_terminal_b'].shift())
            group['trip_id'] = group['block'].cumsum()
            
            for trip_id, trip in group.groupby('trip_id'):
                if len(trip) < 10: continue # Ignora segmentos muito curtos
                
                start_in_a = trip.iloc[0]['at_terminal_a']
                end_in_b = trip.iloc[-1]['at_terminal_b']
                
                start_in_b = trip.iloc[0]['at_terminal_b']
                end_in_a = trip.iloc[-1]['at_terminal_a']
                
                # É uma viagem de Ida se começa em A e não está em B, e termina em B
                if start_in_a and not start_in_b and end_in_b:
                    gdf_ida_list.append(trip)
                # É uma viagem de Volta se começa em B e não está em A, e termina em A
                elif start_in_b and not start_in_a and end_in_a:
                    gdf_volta_list.append(trip)
        
        if not gdf_ida_list or not gdf_volta_list:
            print("  ❌ Falha ao classificar viagens de Ida e Volta. Abortando."); return

        gdf_ida = pd.concat(gdf_ida_list)
        gdf_volta = pd.concat(gdf_volta_list)
        print(f"  ✔️ {len(gdf_ida)} pontos de Ida, {len(gdf_volta)} pontos de Volta classificados.")

        # Etapa 4: Geração da Rota
        print("4. Gerando rotas de referência...")
        route_ida = self._create_reference_route(gdf_ida)
        route_volta = self._create_reference_route(gdf_volta)
        
        self._delete_existing_route(linha_id)
        self._save_route_to_db(linha_id, 'Ida', route_ida)
        self._save_route_to_db(linha_id, 'Volta', route_volta)
        
        self._plot_validation_map(gdf_ida, gdf_volta, route_ida, route_volta, linha_id)


if __name__ == '__main__':
    linhas_para_processar = ['639', '422', '455']
    processor = RouteProcessor(DB_CONFIG)
    processor.connect()
    if processor.conn:
        try:
            # Não precisamos mais das colunas no banco, a classificação é em memória
            processor.setup_routes_table() 
            for linha in linhas_para_processar:
                processor.processar_linha(linha)
        finally:
            processor.close()
    print("\n\n" + "="*50); print("🎉 Processo de criação de rotas concluído!"); print("="*50)
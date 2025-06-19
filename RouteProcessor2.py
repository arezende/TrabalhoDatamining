import geopandas as gpd
import pandas as pd
import numpy as np
import psycopg2
from shapely.geometry import Point, LineString, MultiPoint
from sklearn.cluster import DBSCAN
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
        self.terminal_A = None
        self.terminal_B = None

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
        sql = f"SELECT id, ordem, datahora_servidor, velocidade, geom FROM pontos_gps_treino WHERE linha = %s AND EXTRACT(HOUR FROM datahora_servidor) BETWEEN 8 AND 22 ORDER BY ordem, datahora_servidor;"
        gdf = gpd.read_postgis(sql, self.conn, params=[linha_id], geom_col='geom')
        if gdf.empty:
            print(f"  ⚠️ Nenhum dado encontrado."); return None
        print(f"  ✔️ {len(gdf)} pontos carregados.")
        return gdf

    def _interpolate_tunnels(self, gdf_trip, route_guide):
        if route_guide is None or gdf_trip.empty:
            return gdf_trip
            
        gdf_proj = gdf_trip.to_crs(31983)
        route_proj = gpd.GeoSeries([route_guide], crs=4326).to_crs(31983).iloc[0]
        
        gdf_proj['time_diff'] = gdf_proj['datahora_servidor'].diff().dt.total_seconds().fillna(0)
        gdf_proj['dist_diff'] = gdf_proj.geometry.distance(gdf_proj.geometry.shift()).fillna(0)
        gdf_proj['speed_kmh'] = (gdf_proj['dist_diff'] / gdf_proj['time_diff'] * 3.6).replace([np.inf, -np.inf], 0).fillna(0)

        jump_indices = gdf_proj[(gdf_proj['speed_kmh'] > 120) & (gdf_proj['dist_diff'] > 400)].index
        
        points_to_add = []
        for idx in jump_indices:
            p1_idx = gdf_proj.index.get_loc(idx) - 1
            if p1_idx < 0: continue
            
            p1 = gdf_proj.iloc[p1_idx]
            p2 = gdf_proj.loc[idx]
            
            _, snap_p1 = nearest_points(route_proj, p1.geometry)
            _, snap_p2 = nearest_points(route_proj, p2.geometry)
            
            dist_p1_on_route = route_proj.project(snap_p1)
            dist_p2_on_route = route_proj.project(snap_p2)

            if dist_p1_on_route >= dist_p2_on_route: continue # Movimento para trás, ignora

            # Extrai o segmento do túnel
            try:
                # Criar um LineString com apenas os dois pontos de projeção
                segment_cutter = LineString([snap_p1, snap_p2])
                # Dividir a rota principal usando os pontos
                split_routes = split(route_proj, segment_cutter.centroid)
                # O segmento do túnel é a parte da rota entre os pontos
                tunnel_segment_candidates = [s for s in split(route_proj, MultiPoint([snap_p1, snap_p2])).geoms]
                if not tunnel_segment_candidates: continue
                # Pega o segmento que conecta os dois pontos projetados
                tunnel_segment = min([s for s in tunnel_segment_candidates if s.intersects(segment_cutter)], key=lambda x: x.length)
            except Exception:
                continue
            
            # Interpola novos pontos
            num_points_to_interpolate = int(p2.time_diff / 30) # Um ponto a cada 30s
            for i in range(1, num_points_to_interpolate):
                fraction = i / num_points_to_interpolate
                interp_point = tunnel_segment.interpolate(fraction, normalized=True)
                interp_time = p1.datahora_servidor + pd.to_timedelta(i * 30, unit='s')
                # Adiciona o ponto com todos os dados necessários
                points_to_add.append({'ordem': p1.ordem, 'datahora_servidor': interp_time, 'geom': interp_point, 'velocidade': 50})

        if points_to_add:
            gdf_new_points = gpd.GeoDataFrame(points_to_add, geometry='geom', crs=31983)
            gdf_filled = pd.concat([gdf_proj, gdf_new_points], ignore_index=True).sort_values('datahora_servidor')
            return gdf_filled.to_crs(4326)
        
        return gdf_trip
        
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
        print("5. Gerando mapa de validação...")
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
        if self.terminal_A and self.terminal_B:
            folium.Marker([self.terminal_A.y, self.terminal_A.x], popup="Terminal A", icon=folium.Icon(color='green')).add_to(m)
            folium.Marker([self.terminal_B.y, self.terminal_B.x], popup="Terminal B", icon=folium.Icon(color='purple')).add_to(m)
        folium.LayerControl().add_to(m)
        m.save(f'mapa_validacao_final_{linha_id}.html'); print(f"  ✔️ Mapa salvo.")

    def processar_linha(self, linha_id):
        print("\n" + "="*50); print(f"🚀 Processando Linha: {linha_id}")
        gdf = self._fetch_data(linha_id)
        if gdf is None: return

        print("  - Removendo outliers com DBSCAN...")
        coords_rad = np.radians(gdf[['geom']].apply(lambda r: (r.geom.y, r.geom.x), axis=1).tolist())
        db = DBSCAN(eps=100/6371000, min_samples=10, metric='haversine').fit(coords_rad)
        gdf_clean = gdf[gdf['cluster'] != -1].copy()
        
        print("2. Detectando paradas significativas...")
        stop_points, garage_points_ids = [], []
        gdf_proj = gdf_clean.to_crs(31983)
        for ordem, group in gdf_proj.groupby('ordem'):
            group = group.sort_values('datahora_servidor')
            group['time_diff'] = group['datahora_servidor'].diff().dt.total_seconds()
            group['dist_diff'] = group.geometry.distance(group.geometry.shift())
            group['is_stopped'] = (group['dist_diff'] < 15) & (group['time_diff'] < 120)
            group['stop_block'] = (group['is_stopped'] != group['is_stopped'].shift()).cumsum()
            for _, stop_block in group[group['is_stopped']].groupby('stop_block'):
                duration = stop_block['time_diff'].sum()
                if 300 <= duration <= 2400: stop_points.append(stop_block)
                elif duration > 2400: garage_points_ids.extend(stop_block.id.tolist())
        
        if not stop_points: print("  ❌ Nenhum ponto de parada (5-40 min) encontrado."); return
        gdf_stops = gpd.GeoDataFrame(pd.concat(stop_points), geometry='geom', crs=31983)

        db_terminals = DBSCAN(eps=500, min_samples=10).fit(np.array(gdf_stops.geometry.apply(lambda p: (p.x, p.y)).tolist()))
        gdf_stops['terminal_cluster'] = db_terminals.labels_
        cluster_counts = gdf_stops[gdf_stops['terminal_cluster'] != -1]['terminal_cluster'].value_counts()
        if len(cluster_counts) < 2: print("  ❌ Não foi possível identificar 2 terminais."); return
        
        top_clusters = cluster_counts.head(2).index.tolist()
        self.terminal_A = gpd.GeoSeries(MultiPoint(gdf_stops[gdf_stops['terminal_cluster'] == top_clusters[0]].geometry.tolist()).centroid, crs=31983).to_crs(4326).iloc[0]
        self.terminal_B = gpd.GeoSeries(MultiPoint(gdf_stops[gdf_stops['terminal_cluster'] == top_clusters[1]].geometry.tolist()).centroid, crs=31983).to_crs(4326).iloc[0]
        if self.terminal_A.x > self.terminal_B.x: self.terminal_A, self.terminal_B = self.terminal_B, self.terminal_A
        print("  - Terminais A e B definidos.")

        print("3. Classificando sentidos e corrigindo túneis...")
        gdf_clean = gdf_clean[~gdf_clean.id.isin(garage_points_ids)]
        gdf_clean['dist_a'] = gdf_clean.geometry.distance(self.terminal_A)
        gdf_clean['dist_b'] = gdf_clean.geometry.distance(self.terminal_B)
        gdf_clean['sentido'] = np.where(gdf_clean['dist_a'] < gdf_clean['dist_b'], 'Volta', 'Ida')

        gdf_ida = gdf_clean[gdf_clean['sentido'] == 'Ida']
        gdf_volta = gdf_clean[gdf_clean['sentido'] == 'Volta']

        route_ida_guia = self._create_reference_route(gdf_ida[gdf_ida['velocidade'] >= 20], smoothing_window=50)
        route_volta_guia = self._create_reference_route(gdf_volta[gdf_volta['velocidade'] >= 20], smoothing_window=50)

        corrected_ida_trips = [self._interpolate_tunnels(group, route_ida_guia) for _, group in gdf_ida.groupby('ordem')]
        corrected_volta_trips = [self._interpolate_tunnels(group, route_volta_guia) for _, group in gdf_volta.groupby('ordem')]
        
        gdf_ida_final = gpd.GeoDataFrame(pd.concat(corrected_ida_trips, ignore_index=True), crs=4326) if corrected_ida_trips else gpd.GeoDataFrame()
        gdf_volta_final = gpd.GeoDataFrame(pd.concat(corrected_volta_trips, ignore_index=True), crs=4326) if corrected_volta_trips else gpd.GeoDataFrame()
        print(f"  ✔️ {len(gdf_ida_final)} pontos de Ida, {len(gdf_volta_final)} pontos de Volta (após correção).")

        print("4. Gerando rotas de referência finais...")
        final_route_ida = self._create_reference_route(gdf_ida_final)
        final_route_volta = self._create_reference_route(gdf_volta_final)

        self._delete_existing_route(linha_id)
        self._save_route_to_db(linha_id, 'Ida', final_route_ida)
        self._save_route_to_db(linha_id, 'Volta', final_route_volta)
        
        self._plot_validation_map(gdf_ida_final, gdf_volta_final, final_route_ida, final_route_volta, linha_id)

if __name__ == '__main__':
    linhas_para_processar = ['455', '639', '422']
    processor = RouteProcessor(DB_CONFIG)
    processor.connect()
    if processor.conn:
        try:
            processor.setup_routes_table()
            for linha in linhas_para_processar:
                processor.processar_linha(linha)
        finally:
            processor.close()
    print("\n\n" + "="*50); print("🎉 Processo de criação de rotas concluído!"); print("="*50)
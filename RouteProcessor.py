import geopandas as gpd
import pandas as pd
import numpy as np
import psycopg2
from psycopg2 import sql
from psycopg2.extras import Json
import hdbscan
from shapely.geometry import Point, LineString, MultiPoint
import folium
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
from sklearn.cluster import DBSCAN
from datetime import datetime
import warnings
from scipy.interpolate import make_interp_spline
import traceback

warnings.filterwarnings('ignore', 'UserWarning', module='geopandas')
pd.options.mode.chained_assignment = None
sns.set(style="whitegrid")

# --- CONFIGURAÇÕES - AJUSTE AQUI ---
DB_CONFIG = {
    'user': 'postgres',
    'password': 'P@ssw0rd123',
    'host': '35.247.199.127',
    'port': '5432',
    'dbname': 'onibus_db'
}

class AdvancedRouteProcessor:
    def __init__(self, db_params):
        self.db_params = db_params
        self.conn = None
        self.terminal_A = None
        self.terminal_B = None
        self.metrics = {
            'linha': [],
            'sentido': [],
            'completeness': [],
            'f1_score': [],
            'coverage_ratio': [],
            'processing_time': []
        }
        self.temporal_patterns = pd.DataFrame()

    def connect(self):
        try:
            self.conn = psycopg2.connect(**self.db_params, client_encoding='latin1')
            print("✅ Conexão estabelecida.")
        except Exception as e:
            print(f"❌ Erro ao conectar: {e}")

    def close(self):
        if self.conn:
            self.conn.close()
            print("\n🔌 Conexão fechada.")
            
    def setup_routes_table(self):
        sql = """
        CREATE TABLE IF NOT EXISTS rotas_referencia (
            id SERIAL PRIMARY KEY,
            linha VARCHAR(20) NOT NULL,
            sentido VARCHAR(10) NOT NULL,
            geom GEOMETRY(LineString, 4326),
            metrics JSONB,
            UNIQUE(linha, sentido)
        );
        
        CREATE TABLE IF NOT EXISTS temporal_patterns (
            linha VARCHAR(20) NOT NULL,
            hora INTEGER NOT NULL,
            avg_speed FLOAT,
            num_veiculos INTEGER,
            med_dist FLOAT,
            PRIMARY KEY (linha, hora)
        );
        """
        with self.conn.cursor() as cursor:
            cursor.execute(sql)
            self.conn.commit()
        print("🔧 Tabelas otimizadas criadas.")

    def _calculate_bearing(self, point1, point2):
        """Calcula o ângulo de direção entre dois pontos."""
        lon1, lat1 = np.radians(point1.x), np.radians(point1.y)
        lon2, lat2 = np.radians(point2.x), np.radians(point2.y)
        
        dlon = lon2 - lon1
        x = np.sin(dlon) * np.cos(lat2)
        y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
        
        bearing = np.arctan2(x, y)
        return np.degrees(bearing) % 360

    def _create_reference_route(self, gdf_direction, eps=15):
        if gdf_direction is None or gdf_direction.empty or len(gdf_direction) < 2:
            return None
            
        print(f"  - Criando rota com HDBSCAN (eps={eps}m)...")
        coords = np.array([list(p.coords)[0] for p in gdf_direction.geometry])
        
        # Clusterização com HDBSCAN
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=50,
            min_samples=10,
            cluster_selection_epsilon=eps/6371000,
            metric='haversine'
        ).fit(np.radians(coords))
        
        labels = clusterer.labels_
        unique_labels = set(labels)
        
        if -1 in unique_labels: 
            unique_labels.remove(-1)
            
        if not unique_labels:
            return None
            
        # Selecionar o maior cluster
        largest_cluster_label = max(unique_labels, key=lambda x: np.sum(labels == x))
        route_indices = np.where(labels == largest_cluster_label)[0]
        route_coords = coords[route_indices]
        
        if len(route_coords) < 2:
            return None

        # Criar linha suavizada
        smooth_route = LineString(route_coords)
        return gpd.GeoSeries([smooth_route], crs=4326).iloc[0]

    def _calculate_metrics(self, gdf, route_geom):
        """Calcula métricas de qualidade para a rota gerada."""
        if route_geom is None or gdf is None or gdf.empty:
            return {'completeness': 0, 'f1_score': 0, 'coverage_ratio': 0}
        
        # 1. Completeness: % de pontos próximos à rota
        gdf_proj = gdf.to_crs(epsg=31983)
        route_proj = gpd.GeoSeries([route_geom], crs=4326).to_crs(epsg=31983).iloc[0]
        buffer = route_proj.buffer(50)  # 50m buffer
        
        within_buffer = gdf_proj.geometry.within(buffer)
        completeness = within_buffer.mean()
        
        # 2. Spatial F1-Score
        sample_points = [route_proj.interpolate(i, normalized=True) for i in np.linspace(0, 1, 100)]
        y_true = np.ones(len(gdf_proj))
        y_pred = np.array([1 if p.within(buffer) else 0 for p in gdf_proj.geometry])
        
        # 3. Coverage Ratio
        coverage_ratio = len(sample_points) / max(1, len(gdf_proj))
        
        return {
            'completeness': completeness,
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'coverage_ratio': coverage_ratio
        }

    def _analyze_temporal_patterns(self, gdf, linha_id):
        """Analisa padrões temporais de operação."""
        if gdf is None or gdf.empty:
            return
            
        print("  📊 Analisando padrões temporais...")
        gdf['hora'] = gdf['datahora_servidor'].dt.hour
        
        # Converter para projeção métrica para cálculos precisos
        gdf_metric = gdf.to_crs(epsg=31983)
        gdf_metric['dist'] = gdf_metric.geometry.distance(gdf_metric.geometry.shift())
        
        patterns = gdf_metric.groupby('hora').agg(
            avg_speed=('velocidade', 'mean'),
            num_veiculos=('ordem', pd.Series.nunique),
            med_dist=('dist', 'median')
        ).reset_index()
        
        patterns['linha'] = linha_id
        
        # Salvar no banco de dados
        with self.conn.cursor() as cursor:
            for _, row in patterns.iterrows():
                cursor.execute("""
                    INSERT INTO temporal_patterns 
                    (linha, hora, avg_speed, num_veiculos, med_dist)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (linha, hora) DO UPDATE SET
                    avg_speed = EXCLUDED.avg_speed,
                    num_veiculos = EXCLUDED.num_veiculos,
                    med_dist = EXCLUDED.med_dist;
                """, (row['linha'], row['hora'], row['avg_speed'], 
                      row['num_veiculos'], row['med_dist']))
            self.conn.commit()
        
        # Atualizar dataframe interno
        self.temporal_patterns = pd.concat([self.temporal_patterns, patterns])
        
        # Gerar visualização
        self._plot_temporal_patterns(patterns, linha_id)

    def _plot_temporal_patterns(self, patterns, linha_id):
        """Gera visualização dos padrões temporais."""
        plt.figure(figsize=(12, 8))
        
        plt.subplot(311)
        sns.lineplot(x='hora', y='avg_speed', data=patterns, marker='o')
        plt.title(f'Velocidade Média por Hora - Linha {linha_id}')
        plt.ylabel('Km/h')
        
        plt.subplot(312)
        sns.lineplot(x='hora', y='num_veiculos', data=patterns, marker='o', color='green')
        plt.title('Número de Veículos em Operação')
        plt.ylabel('Quantidade')
        
        plt.subplot(313)
        sns.lineplot(x='hora', y='med_dist', data=patterns, marker='o', color='red')
        plt.title('Distância Média entre Pontos')
        plt.ylabel('Metros')
        plt.xlabel('Hora do Dia')
        
        plt.tight_layout()
        plt.savefig(f'temporal_patterns_{linha_id}.png')
        plt.close()
        print(f"  ✔️ Padrões temporais salvos: temporal_patterns_{linha_id}.png")

    def _detect_terminals(self, gdf_stops):
        """Detecta terminais usando HDBSCAN com parâmetros adaptativos."""
        if gdf_stops is None or gdf_stops.empty:
            return False
            
        print("  🔍 Detectando terminais com HDBSCAN...")
        coords = np.array([(p.x, p.y) for p in gdf_stops.geometry])
        
        # Clusterização adaptativa
        min_cluster_size = max(20, int(len(gdf_stops)*0.05))
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            cluster_selection_epsilon=150
        ).fit(coords)
        
        gdf_stops['terminal_cluster'] = clusterer.labels_
        cluster_counts = gdf_stops[gdf_stops['terminal_cluster'] != -1]['terminal_cluster'].value_counts()
        
        if len(cluster_counts) < 2:
            return False
            
        top_clusters = cluster_counts.head(2).index.tolist()
        terminal_A = MultiPoint(
            gdf_stops[gdf_stops['terminal_cluster'] == top_clusters[0]].geometry.tolist()
        ).centroid
        terminal_B = MultiPoint(
            gdf_stops[gdf_stops['terminal_cluster'] == top_clusters[1]].geometry.tolist()
        ).centroid
        
        if terminal_A.distance(terminal_B) < 1000:
            return False
            
        terminals_wgs84 = gpd.GeoSeries([terminal_A, terminal_B], crs=gdf_stops.crs).to_crs(4326)
        self.terminal_A, self.terminal_B = terminals_wgs84.iloc[0], terminals_wgs84.iloc[1]
        
        # Ordenar de oeste para leste
        if self.terminal_A.x > self.terminal_B.x:
            self.terminal_A, self.terminal_B = self.terminal_B, self.terminal_A
            
        print(f"  - Terminal A: ({self.terminal_A.x:.4f}, {self.terminal_A.y:.4f})")
        print(f"  - Terminal B: ({self.terminal_B.x:.4f}, {self.terminal_B.y:.4f})")
        return True

    def _segment_trips(self, gdf_vehicle):
        """Segmenta viagens usando análise de direção com bearing."""
        if self.terminal_A is None or self.terminal_B is None:
            return []
            
        gdf_vehicle = gdf_vehicle.sort_values('datahora_servidor')
        gdf_vehicle_proj = gdf_vehicle.to_crs(epsg=31983)
        terminal_a_proj = gpd.GeoSeries([self.terminal_A], crs=4326).to_crs(epsg=31983).iloc[0]
        terminal_b_proj = gpd.GeoSeries([self.terminal_B], crs=4326).to_crs(epsg=31983).iloc[0]
        
        # Calcular direção para terminais
        terminal_direction = self._calculate_bearing(self.terminal_A, self.terminal_B)
        
        trips = []
        current_trip = []
        in_terminal_zone = False
        
        for idx, row in gdf_vehicle_proj.iterrows():
            current_trip.append(row)
            
            # Acessar geometria pela coluna
            geom = row['geom']
            dist_a = geom.distance(terminal_a_proj)
            dist_b = geom.distance(terminal_b_proj)
            
            if min(dist_a, dist_b) < 300:
                if not in_terminal_zone and len(current_trip) > 10:
                    # Classificar viagem usando bearing
                    trip_gdf = gpd.GeoDataFrame(current_trip, geometry='geom', crs=31983)
                    first_point = trip_gdf.iloc[0]['geom']
                    last_point = trip_gdf.iloc[-1]['geom']
                    
                    if first_point and last_point:
                        movement_direction = self._calculate_bearing(
                            Point(first_point.x, first_point.y), 
                            Point(last_point.x, last_point.y)
                        )
                        
                        # Calcular diferença angular
                        angle_diff = abs(movement_direction - terminal_direction) % 360
                        if angle_diff > 180:
                            angle_diff = 360 - angle_diff
                        
                        sentido = 'Ida' if angle_diff < 45 else 'Volta'
                        trip_gdf['sentido'] = sentido
                        trips.append(trip_gdf.to_crs(4326))
                    
                    current_trip = []
                in_terminal_zone = True
            else:
                in_terminal_zone = False
        
        # Adicionar última viagem
        if len(current_trip) > 10:
            trip_gdf = gpd.GeoDataFrame(current_trip, geometry='geom', crs=31983)
            trips.append(trip_gdf.to_crs(4326))
        
        print(f"  - Segmentadas {len(trips)} viagens para ordem {gdf_vehicle['ordem'].iloc[0]}")
        return trips

    def _create_base_map(self, gdf, title, filename):
        """Cria um mapa base com todos os pontos."""
        if gdf is None or gdf.empty:
            print(f"  ⚠️ Nenhum dado para criar mapa: {title}")
            return
            
        # Usar union_all() em vez de unary_union
        center = gdf.geometry.union_all().centroid
        m = folium.Map(location=[center.y, center.x], zoom_start=13, tiles='cartodbpositron')
        
        # Adicionar todos os pontos
        for idx, row in gdf.sample(min(1000, len(gdf))).iterrows():
            # Acessar a coluna 'geom' que contém a geometria
            point = row['geom']
            folium.CircleMarker(
                location=[point.y, point.x],
                radius=3,
                color='blue',
                fill=True,
                fill_opacity=0.7
            ).add_to(m)
        
        # Adicionar título
        folium.Marker(
            location=[center.y, center.x],
            icon=folium.DivIcon(html=f"<div style='font-size: 16pt; color: red'>{title}</div>")
        ).add_to(m)
        
        m.save(filename)
        print(f"  ✔️ Mapa base salvo: {filename}")

    def _create_stops_map(self, gdf_stops, filename):
        """Cria mapa com pontos de parada detectados."""
        if gdf_stops is None or gdf_stops.empty:
            print(f"  ⚠️ Nenhum ponto de parada para criar mapa")
            return
            
        center = gdf_stops.geometry.union_all().centroid
        m = folium.Map(location=[center.y, center.x], zoom_start=13, tiles='cartodbpositron')
        
        # Adicionar pontos de parada
        for idx, row in gdf_stops.iterrows():
            point = row['geom']  # Acessar geometria pela coluna
            folium.CircleMarker(
                location=[point.y, point.x],
                radius=5,
                color='red',
                fill=True,
                fill_opacity=0.7,
                popup=f"Ordem: {row['ordem']}<br>Duração: {row['time_diff']/60:.1f} min"
            ).add_to(m)
        
        # Adicionar terminais se existirem
        if self.terminal_A and self.terminal_B:
            folium.Marker(
                [self.terminal_A.y, self.terminal_A.x],
                popup="Terminal A",
                icon=folium.Icon(color='green', icon='bus', prefix='fa')
            ).add_to(m)
            
            folium.Marker(
                [self.terminal_B.y, self.terminal_B.x],
                popup="Terminal B",
                icon=folium.Icon(color='purple', icon='bus', prefix='fa')
            ).add_to(m)
        
        m.save(filename)
        print(f"  ✔️ Mapa de paradas salvo: {filename}")

    def _plot_validation_map(self, gdf_ida, gdf_volta, route_ida, route_volta, linha_id):
        """Gera mapa interativo de validação com mais camadas."""
        print("4. Gerando mapa de validação avançado...")
        
        # Determinar centro do mapa
        if gdf_ida is not None and not gdf_ida.empty:
            center_point = gdf_ida.geometry.union_all().centroid
        elif gdf_volta is not None and not gdf_volta.empty:
            center_point = gdf_volta.geometry.union_all().centroid
        else:
            print("  - Nenhum dado para plotar.")
            return
            
        m = folium.Map(location=[center_point.y, center_point.x], zoom_start=13, tiles='cartodbpositron')
        
        # Heatmap de densidade
        if gdf_ida is not None and not gdf_ida.empty:
            heat_data_ida = [[p.y, p.x] for p in gdf_ida.geometry]
            folium.plugins.HeatMap(heat_data_ida, name='Densidade Ida', radius=10).add_to(m)
        
        if gdf_volta is not None and not gdf_volta.empty:
            heat_data_volta = [[p.y, p.x] for p in gdf_volta.geometry]
            folium.plugins.HeatMap(heat_data_volta, name='Densidade Volta', radius=10, gradient={0.4: 'blue', 0.65: 'lime', 1: 'red'}).add_to(m)
        
        # Rotas geradas
        if route_ida:
            folium.GeoJson(
                route_ida,
                name="Rota Ida (Gerada)",
                style_function=lambda x: {'color': 'blue', 'weight': 5, 'opacity': 0.8}
            ).add_to(m)
        
        if route_volta:
            folium.GeoJson(
                route_volta,
                name="Rota Volta (Gerada)",
                style_function=lambda x: {'color': 'red', 'weight': 5, 'opacity': 0.8}
            ).add_to(m)
        
        # Terminais
        if self.terminal_A and self.terminal_B:
            folium.Marker(
                [self.terminal_A.y, self.terminal_A.x],
                popup="Terminal A",
                icon=folium.Icon(color='green', icon='bus', prefix='fa')
            ).add_to(m)
            
            folium.Marker(
                [self.terminal_B.y, self.terminal_B.x],
                popup="Terminal B",
                icon=folium.Icon(color='purple', icon='bus', prefix='fa')
            ).add_to(m)
        
        # Controles de camada
        folium.LayerControl(collapsed=False).add_to(m)
        m.save(f'mapa_validacao_avancado_{linha_id}.html')
        print(f"  ✔️ Mapa avançado salvo: mapa_validacao_avancado_{linha_id}.html")

    def _save_route_to_db(self, linha_id, sentido, route_geom, metrics):
        if route_geom is None:
            print(f"    ⚠️ Rota '{sentido}' vazia, não salva.")
            return
            
        metrics_json = {
            'completeness': metrics['completeness'],
            'f1_score': metrics['f1_score'],
            'coverage_ratio': metrics['coverage_ratio'],
            'processing_time': metrics['processing_time']
        }
        
        query = sql.SQL("""
            INSERT INTO rotas_referencia (linha, sentido, geom, metrics)
            VALUES (%s, %s, ST_GeomFromText(%s, 4326), %s)
            ON CONFLICT (linha, sentido) DO UPDATE SET
                geom = EXCLUDED.geom,
                metrics = EXCLUDED.metrics;
        """)
        
        with self.conn.cursor() as cursor:
            cursor.execute(query, (linha_id, sentido, route_geom.wkt, Json(metrics_json)))
            self.conn.commit()
        print(f"    ✔️ Rota '{sentido}' salva com métricas.")

    def _plot_speed_profile(self, gdf, route_geom, linha_id, sentido):
        """Gera perfil de velocidade ao longo da rota."""
        if route_geom is None or gdf is None or gdf.empty:
            return
            
        # Projetar para sistema métrico
        gdf_metric = gdf.to_crs(epsg=31983)
        route_metric = gpd.GeoSeries([route_geom], crs=4326).to_crs(epsg=31983).iloc[0]
        
        # Calcular distância ao longo da rota
        gdf_metric['dist_along'] = gdf_metric.geometry.apply(
            lambda p: route_metric.project(p) if route_metric.distance(p) < 100 else np.nan
        )
        
        gdf_metric = gdf_metric.dropna(subset=['dist_along'])
        
        if gdf_metric.empty:
            return
            
        # Agrupar por segmentos
        bins = np.linspace(0, route_metric.length, 50)
        gdf_metric['segment'] = pd.cut(gdf_metric['dist_along'], bins, include_lowest=True)
        
        # Adicionar observed=False para evitar warnings
        profile = gdf_metric.groupby('segment', observed=False).agg(
            avg_speed=('velocidade', 'mean'),
            count=('velocidade', 'count')
        ).reset_index()
        
        profile['mid_dist'] = profile['segment'].apply(lambda x: x.mid)
        
        # Plot
        plt.figure(figsize=(14, 6))
        plt.scatter(profile['mid_dist'], profile['avg_speed'], 
                   s=profile['count']/10, alpha=0.7, c='blue')
        
        # Suavizar
        if len(profile) >= 4:  # Requer pelo menos 4 pontos para spline
            try:
                xnew = np.linspace(profile['mid_dist'].min(), profile['mid_dist'].max(), 300)
                spl = make_interp_spline(profile['mid_dist'], profile['avg_speed'], k=3)
                y_smooth = spl(xnew)
                plt.plot(xnew, y_smooth, 'r-', linewidth=2)
            except Exception as e:
                print(f"    ⚠️ Erro ao suavizar perfil: {e}")
        
        plt.title(f'Perfil de Velocidade - Linha {linha_id} ({sentido})')
        plt.xlabel('Distância ao Longo da Rota (metros)')
        plt.ylabel('Velocidade Média (km/h)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # CORREÇÃO: Usar savefig() em vez de save()
        plt.savefig(f'speed_profile_{linha_id}_{sentido}.png')
        plt.close()
        print(f"  ✔️ Perfil de velocidade salvo: speed_profile_{linha_id}_{sentido}.png")

    def _fetch_data(self, linha_id):
        print(f"1. Buscando dados da linha '{linha_id}'...")
        query = sql.SQL("""
            SELECT id, ordem, datahora_servidor, velocidade, geom
            FROM pontos_gps_treino
            WHERE linha = %s
            AND EXTRACT(HOUR FROM datahora_servidor) BETWEEN 5 AND 23
            ORDER BY ordem, datahora_servidor;
        """)
        try:
            # Usar read_postgis com SQL parametrizado
            gdf = gpd.read_postgis(
                query.as_string(self.conn), 
                self.conn, 
                params=[linha_id], 
                geom_col='geom'
            )
            
            if gdf.empty:
                print(f"  ⚠️ Nenhum dado encontrado para linha {linha_id}")
                return None
            
            print(f"  ✔️ {len(gdf)} pontos carregados.")
            
            # Salvar mapa de todos os pontos
            self._create_base_map(gdf, f"Pontos GPS - Linha {linha_id}", f"mapa_base_{linha_id}.html")
            
            return gdf
        except Exception as e:
            print(f"  ❌ Erro ao carregar dados: {e}")
            traceback.print_exc()
            return None

    def processar_linha(self, linha_id):
        start_time = datetime.now()
        print("\n" + "="*50)
        print(f"🚀 Processando Linha: {linha_id}")
        print(f"⏰ Início: {start_time.strftime('%H:%M:%S')}")
        print("="*50)
        
        # Etapa 1: Carregar e pré-processar dados
        gdf = self._fetch_data(linha_id)
        if gdf is None:
            return
        
        # Pré-processamento: remover outliers
        print("\n🔍 1. Detecção de Outliers")
        print("  - Removendo outliers com DBSCAN...")
        coords_rad = np.radians([list(p.coords)[0] for p in gdf.geometry])
        db = DBSCAN(eps=150/6371000, min_samples=5, metric='haversine').fit(coords_rad)
        gdf['cluster'] = db.labels_
        
        # Filtrar dados
        gdf_clean = gdf[(gdf['cluster'] != -1) & (gdf['velocidade'] <= 100)].copy()
        print(f"  - {len(gdf) - len(gdf_clean)} outliers removidos.")
        
        # Etapa 2: Detecção de Terminais e Paradas
        print("\n🔍 2. Detecção de Terminais e Paradas")
        
        # Converter para projeção métrica
        gdf_clean_proj = gdf_clean.to_crs(epsg=31983)
        
        # Detectar paradas significativas (5-40 minutos)
        stop_points = []
        garage_points_ids = []
        
        for ordem, group in gdf_clean_proj.groupby('ordem'):
            group = group.sort_values('datahora_servidor')
            group['time_diff'] = group['datahora_servidor'].diff().dt.total_seconds()
            group['dist_diff'] = group.geometry.distance(group.geometry.shift())

            group['is_stopped'] = (group['dist_diff'] < 15) & (group['time_diff'] < 120)
            group['stop_block'] = (group['is_stopped'] != group['is_stopped'].shift()).cumsum()

            for block_id, stop_block in group[group['is_stopped']].groupby('stop_block'):
                duration = stop_block['time_diff'].sum()
                if 300 <= duration <= 2400:  # 5 a 40 minutos
                    stop_points.append(stop_block)
                elif duration > 2400:  # mais de 40 minutos -> garagem
                    garage_points_ids.extend(stop_block.id.tolist())

        if not stop_points:
            print("  ❌ Nenhum ponto de parada significativo encontrado. Abortando.")
            return

        df_stops = pd.concat(stop_points)
        gdf_stops = gpd.GeoDataFrame(df_stops, geometry='geom', crs=31983)
        print(f"  - {len(garage_points_ids)} pontos de garagem identificados.")
        print(f"  - {len(gdf_stops)} pontos em paradas de terminal encontrados.")
        
        # Salvar mapa de paradas
        self._create_stops_map(gdf_stops, f"mapa_paradas_{linha_id}.html")
        
        # Detectar terminais usando HDBSCAN
        if not self._detect_terminals(gdf_stops):
            print("  ❌ Falha na detecção de terminais. Abortando.")
            return
            
        # Etapa 3: Segmentação e Classificação de Viagens
        print("\n🛣️ 3. Segmentação de Viagens")
        # Remover pontos de garagem
        gdf_clean = gdf_clean[~gdf_clean.id.isin(garage_points_ids)]
        gdf_fast = gdf_clean[gdf_clean['velocidade'] >= 20].copy()
        
        all_ida_trips = []
        all_volta_trips = []
        
        for ordem, group in gdf_fast.groupby('ordem'):
            trips = self._segment_trips(group)
            for trip in trips:
                if 'sentido' in trip.columns:
                    if trip['sentido'].iloc[0] == 'Ida':
                        all_ida_trips.append(trip)
                    else:
                        all_volta_trips.append(trip)
        
        gdf_ida = gpd.GeoDataFrame(pd.concat(all_ida_trips, ignore_index=True), 
                                  geometry='geom', crs=4326) if all_ida_trips else None
        gdf_volta = gpd.GeoDataFrame(pd.concat(all_volta_trips, ignore_index=True), 
                                    geometry='geom', crs=4326) if all_volta_trips else None
        
        print(f"  ✔️ {len(gdf_ida) if gdf_ida is not None else 0} pontos de Ida, "
              f"{len(gdf_volta) if gdf_volta is not None else 0} pontos de Volta")
        
        # Salvar mapas de direções
        if gdf_ida is not None and not gdf_ida.empty:
            self._create_base_map(gdf_ida, f"Pontos Ida - Linha {linha_id}", f"mapa_ida_{linha_id}.html")
        if gdf_volta is not None and not gdf_volta.empty:
            self._create_base_map(gdf_volta, f"Pontos Volta - Linha {linha_id}", f"mapa_volta_{linha_id}.html")
        
        # Etapa 4: Geração de Rotas e Análise
        print("\n📈 4. Geração de Rotas e Análise")
        route_ida = self._create_reference_route(gdf_ida) if gdf_ida is not None and not gdf_ida.empty else None
        route_volta = self._create_reference_route(gdf_volta) if gdf_volta is not None and not gdf_volta.empty else None
        
        # Cálculo de métricas
        metrics_ida = self._calculate_metrics(gdf_ida, route_ida) if gdf_ida is not None else {'completeness': 0, 'f1_score': 0, 'coverage_ratio': 0}
        metrics_volta = self._calculate_metrics(gdf_volta, route_volta) if gdf_volta is not None else {'completeness': 0, 'f1_score': 0, 'coverage_ratio': 0}
        
        # Análise temporal
        self._analyze_temporal_patterns(gdf_clean, linha_id)
        
        # Visualizações
        self._plot_speed_profile(gdf_ida, route_ida, linha_id, 'Ida')
        self._plot_speed_profile(gdf_volta, route_volta, linha_id, 'Volta')
        
        # Salvar rotas com métricas
        processing_time = (datetime.now() - start_time).total_seconds()
        
        if route_ida:
            metrics_ida['processing_time'] = processing_time
            self._save_route_to_db(linha_id, 'Ida', route_ida, metrics_ida)
            
        if route_volta:
            metrics_volta['processing_time'] = processing_time
            self._save_route_to_db(linha_id, 'Volta', route_volta, metrics_volta)
        
        # Mapa de validação
        self._plot_validation_map(gdf_ida, gdf_volta, route_ida, route_volta, linha_id)
        
        # Atualizar métricas para relatório final
        if route_ida:
            self.metrics['linha'].append(linha_id)
            self.metrics['sentido'].append('Ida')
            self.metrics['completeness'].append(metrics_ida['completeness'])
            self.metrics['f1_score'].append(metrics_ida['f1_score'])
            self.metrics['coverage_ratio'].append(metrics_ida['coverage_ratio'])
            self.metrics['processing_time'].append(processing_time)
            
        if route_volta:
            self.metrics['linha'].append(linha_id)
            self.metrics['sentido'].append('Volta')
            self.metrics['completeness'].append(metrics_volta['completeness'])
            self.metrics['f1_score'].append(metrics_volta['f1_score'])
            self.metrics['coverage_ratio'].append(metrics_volta['coverage_ratio'])
            self.metrics['processing_time'].append(processing_time)
            
        print(f"\n⏱️ Tempo total: {processing_time:.2f} segundos")
        
    def generate_final_report(self):
        """Gera relatório final com métricas consolidadas."""
        print("\n📊 Gerando relatório final...")
        df_metrics = pd.DataFrame(self.metrics)
        
        if df_metrics.empty:
            print("  ⚠️ Nenhuma métrica disponível para relatório")
            return
            
        # Salvar métricas
        df_metrics.to_csv('route_metrics.csv', index=False)
        
        # Plotar comparação de algoritmos
        plt.figure(figsize=(12, 8))
        
        plt.subplot(221)
        sns.boxplot(x='sentido', y='completeness', data=df_metrics)
        plt.title('Completude por Sentido')
        
        plt.subplot(222)
        sns.scatterplot(x='processing_time', y='f1_score', hue='linha', 
                        size='coverage_ratio', data=df_metrics, palette='viridis')
        plt.title('Desempenho vs Tempo de Processamento')
        
        plt.subplot(212)
        if not self.temporal_patterns.empty:
            sns.lineplot(x='hora', y='avg_speed', hue='linha', 
                         data=self.temporal_patterns, estimator='median', err_style=None)
            plt.title('Velocidade Média por Hora (Todas as Linhas)')
        
        plt.tight_layout()
        plt.savefig('final_report.png')
        plt.close()
        print("  ✔️ Relatório final salvo: final_report.png")


if __name__ == '__main__':
    linhas_para_processar = ['483', '864', '639', '3', '309']
    
    processor = AdvancedRouteProcessor(DB_CONFIG)
    processor.connect()
    
    if processor.conn:
        try:
            processor.setup_routes_table()
            for linha in linhas_para_processar:
                processor.processar_linha(linha)
                
            processor.generate_final_report()
        except Exception as e:
            print(f"❌ Erro durante o processamento: {e}")
            traceback.print_exc()
        finally:
            processor.close()
    
    print("\n\n" + "="*50)
    print("🎉 Processo de criação de rotas concluído!")
    print("="*50)
    print("📈 Métricas salvas em: route_metrics.csv")
    print("🕒 Padrões temporais salvos no banco de dados")
    print("🗺️ Mapas de validação salvos como HTML")
    print("📊 Relatório final: final_report.png")
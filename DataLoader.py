import os
import zipfile
import json
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime
import io

# --- CONFIGURAÇÕES - AJUSTE AQUI ---
DB_CONFIG = {
    'dbname': 'onibus_db',
    'user': 'postgres',
    'password': 'P@ssw0rd123',
    'host': '172.18.166.112',
    'port': '5432'
}
# Pasta onde estão os arquivos .zip
DADOS_ZIP_DIR = 'dados_onibus'
BATCH_SIZE = 1000
# --- FIM DAS CONFIGURAÇÕES ---


class DataLoader:
    """
    Carrega dados de treino (histórico) e teste (tarefas de previsão)
    para o PostgreSQL/PostGIS a partir de arquivos ZIP.
    """
    def __init__(self, db_params):
        self.db_params = db_params
        self.conn = None

    def connect(self):
        try:
            self.conn = psycopg2.connect(**self.db_params, client_encoding='UTF8')
            print("✅ Conexão com o PostgreSQL bem-sucedida!")
        except psycopg2.OperationalError as e:
            print(f"❌ Erro ao conectar: {e}")
            self.conn = None

    def close(self):
        if self.conn:
            self.conn.close()
            print("\n🔌 Conexão fechada.")

    def setup_tables(self):
        if not self.conn: return

        # Tabela para os dados de treino (histórico completo)
        create_treino_sql = """
        CREATE TABLE IF NOT EXISTS pontos_gps_treino (
            id BIGSERIAL PRIMARY KEY,
            ordem VARCHAR(15) NOT NULL,
            linha VARCHAR(20),
            velocidade INTEGER,
            datahora_servidor TIMESTAMPTZ,
            latitude DOUBLE PRECISION NOT NULL,
            longitude DOUBLE PRECISION NOT NULL,
            geom GEOMETRY(Point, 4326)
        );
        CREATE INDEX IF NOT EXISTS idx_treino_geom ON pontos_gps_treino USING GIST (geom);
        CREATE INDEX IF NOT EXISTS idx_treino_datahora ON pontos_gps_treino (datahora_servidor);
        """

        # Tabela para os dados de teste (as "perguntas" a serem previstas)
        create_teste_sql = """
        CREATE TABLE IF NOT EXISTS pontos_gps_teste (
            id VARCHAR(50) PRIMARY KEY,
            linha VARCHAR(20),
            ordem VARCHAR(15),
            datahora TIMESTAMPTZ,  -- Preenchido se a pergunta for sobre posição
            latitude DOUBLE PRECISION,   -- Preenchido se a pergunta for sobre tempo
            longitude DOUBLE PRECISION,  -- Preenchido se a pergunta for sobre tempo
            geom GEOMETRY(Point, 4326)
        );
        """
        
        with self.conn.cursor() as cursor:
            try:
                print("🔧 Configurando tabelas 'pontos_gps_treino' e 'pontos_gps_teste'...")
                cursor.execute(create_treino_sql)
                cursor.execute(create_teste_sql)
                self.conn.commit()
                print("  ✔️  Estrutura do banco de dados está pronta.")
            except Exception as e:
                self.conn.rollback(); raise e

    def _insert_batch(self, sql_query, data_batch):
        if not data_batch: return 0
        with self.conn.cursor() as cursor:
            try:
                execute_values(cursor, sql_query, data_batch, page_size=len(data_batch))
                self.conn.commit()
                return len(data_batch)
            except Exception as e:
                self.conn.rollback(); print(f"❌ Erro ao inserir lote: {e}")
                return 0

    def _load_treino_data(self, data):
        """Carrega dados para a tabela pontos_gps_treino (arquivos 2024-xx-xx)."""
        sql = """
            INSERT INTO pontos_gps_treino (ordem, linha, velocidade, datahora_servidor, latitude, longitude, geom)
            VALUES %s;
        """
        batch = []
        for ponto in data:
            try:
                if not all(k in ponto for k in ['latitude', 'longitude', 'datahoraservidor']): continue
                ts_servidor_ms = int(ponto['datahoraservidor'])
                datahora = datetime.fromtimestamp(ts_servidor_ms / 1000.0)
                lat = float(str(ponto['latitude']).replace(',', '.'))
                lon = float(str(ponto['longitude']).replace(',', '.'))
                geom_val = f"POINT({lon} {lat})"
                batch.append((
                    ponto.get('ordem'), ponto.get('linha'), int(ponto.get('velocidade', 0)),
                    datahora, lat, lon, geom_val
                ))
            except (ValueError, TypeError, KeyError): continue
        return self._insert_batch(sql, batch)

    def _load_teste_data(self, data):
        """Carrega dados para a tabela pontos_gps_teste (arquivos treino- e teste-)."""
        sql = """
            INSERT INTO pontos_gps_teste (id, linha, ordem, datahora, latitude, longitude, geom)
            VALUES %s ON CONFLICT (id) DO NOTHING;
        """
        batch = []
        for ponto in data:
            try:
                datahora_p, lat_p, lon_p, geom_p = None, None, None, None
                # Caso 1: Prever POSIÇÃO (dado o tempo)
                if 'datahora' in ponto:
                    ts_ms = int(ponto['datahora'])
                    datahora_p = datetime.fromtimestamp(ts_ms / 1000.0)
                # Caso 2: Prever TEMPO (dada a posição)
                elif 'latitude' in ponto and 'longitude' in ponto:
                    lat_p = float(str(ponto['latitude']).replace(',', '.'))
                    lon_p = float(str(ponto['longitude']).replace(',', '.'))
                    geom_p = f"POINT({lon_p} {lat_p})"
                else: continue

                batch.append((
                    ponto['id'], ponto.get('linha'), ponto.get('ordem'),
                    datahora_p, lat_p, lon_p, geom_p
                ))
            except (ValueError, TypeError, KeyError) as e:
                print(f"Erro no ponto de teste {ponto.get('id')}: {e}")
                continue
        return self._insert_batch(sql, batch)

    def run(self, zip_directory):
        """Orquestra o processo de carregamento, lendo de múltiplos arquivos ZIP."""
        self.connect()
        if not self.conn: return
        
        try:
            self.setup_tables()
            zip_files = [f for f in os.listdir(zip_directory) if f.endswith('.zip')]
            print(f"Encontrados {len(zip_files)} arquivos ZIP para processar.")

            for zip_filename in sorted(zip_files):
                zip_filepath = os.path.join(zip_directory, zip_filename)
                print(f"\n▼ Abrindo ZIP: {zip_filename}")
                with zipfile.ZipFile(zip_filepath, 'r') as z:
                    json_filenames = [name for name in z.namelist() if name.endswith('.json')]
                    for json_filename in sorted(json_filenames):
                        print(f"  🔄 Processando JSON: {json_filename}...")
                        with z.open(json_filename) as binary_file:
                            text_stream = io.TextIOWrapper(binary_file, encoding='latin-1')
                            dados_json = json.load(text_stream)

                        if json_filename.__contains__('treino-') or json_filename.__contains__('teste-'):
                            count = self._load_teste_data(dados_json)
                            print(f"    ✔️  {count} pontos de teste inseridos.")
                        elif json_filename.__contains__('2024-'):
                            count = self._load_treino_data(dados_json)
                            print(f"    ✔️  {count} pontos de treino inseridos.")
                        else:
                            print(f"    ⚠️  Arquivo '{json_filename}' ignorado.")
        except Exception as e:
            print(f"\nOcorreu um erro geral: {e}")
        finally:
            self.close()

if __name__ == '__main__':
    loader = DataLoader(DB_CONFIG)
    loader.run(DADOS_ZIP_DIR)
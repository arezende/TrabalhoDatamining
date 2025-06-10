import os
import zipfile
import json
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime

# --- CONFIGURAÇÕES - AJUSTE AQUI ---
DB_CONFIG = {
    'dbname': 'onibus_db',
    'user': 'postgres',        # Ou o usuário que você criou, ex: 'anderson'
    'password': 'P@ssw0rd123', # Senha do usuário. Se for o usuário 'postgres', pode não precisar no WSL.
    'host': '172.18.166.112',       # 'localhost' se o script rodar no mesmo WSL do banco
    'port': '5432'
}
DADOS_DIR = 'dados_onibus' # Pasta onde estão os arquivos .zip
BATCH_SIZE = 1000          # Quantidade de registros para inserir por vez (mais eficiente)
# --- FIM DAS CONFIGURAÇÕES ---

def conectar_db():
    """Tenta conectar ao banco de dados e retorna a conexão."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        print("✅ Conexão com o PostgreSQL bem-sucedida!")
        return conn
    except psycopg2.OperationalError as e:
        print(f"❌ Erro ao conectar ao banco de dados: {e}")
        print("Dicas:")
        print("1. Verifique se o serviço do PostgreSQL está rodando ('sudo systemctl status postgresql').")
        print("2. Confira se as informações em DB_CONFIG (dbname, user, password) estão corretas.")
        return None

def setup_database(conn):
    """Cria a tabela e os índices necessários se eles não existirem."""
    
    # Usamos CREATE TABLE IF NOT EXISTS para evitar erros se a tabela já existir.
    # O mesmo para os índices.
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS pontos_gps (
        id BIGSERIAL PRIMARY KEY,
        ordem VARCHAR(15) NOT NULL,
        linha VARCHAR(20),
        velocidade INTEGER,
        datahora_servidor TIMESTAMPTZ, -- 'TIMESTAMPTZ' é ideal para timestamps
        latitude DOUBLE PRECISION NOT NULL,
        longitude DOUBLE PRECISION NOT NULL,
        -- Coluna de geometria: armazena o ponto no sistema de coordenadas WGS84 (padrão GPS)
        geom GEOMETRY(Point, 4326)
    );
    
    -- Criar um índice espacial na coluna de geometria para acelerar as buscas
    CREATE INDEX IF NOT EXISTS idx_pontos_gps_geom ON pontos_gps USING GIST (geom);

    -- Opcional: Criar um índice na data para buscas rápidas por período
    CREATE INDEX IF NOT EXISTS idx_pontos_gps_datahora ON pontos_gps (datahora_servidor);

    -- Opcional: Criar um índice na ordem do ônibus
    CREATE INDEX IF NOT EXISTS idx_pontos_gps_ordem ON pontos_gps (ordem);
    """
    
    cursor = conn.cursor()
    try:
        print("🔧 Verificando e configurando a estrutura do banco de dados (tabela e índices)...")
        cursor.execute(create_table_sql)
        conn.commit()
        print("  ✔️  Estrutura do banco de dados está pronta.")
    except Exception as e:
        print(f"❌ Erro ao configurar o banco de dados: {e}")
        conn.rollback()
        raise e # Levanta a exceção para parar o script se a DB setup falhar
    finally:
        cursor.close()


def processar_arquivo_zip(filepath, conn):
    """Extrai, lê e insere dados de um único arquivo zip no banco de dados."""
    print(f"\n🔄 Processando arquivo: {os.path.basename(filepath)}...")
    total_pontos_no_arquivo = 0
    
    try:
        with zipfile.ZipFile(filepath, 'r') as z:
            json_filename = z.namelist()[0]
            with z.open(json_filename) as json_file:
                dados_json = json.load(json_file)
                pontos_para_inserir = []
                
                for ponto in dados_json:
                    try:
                        ts_servidor_ms = int(ponto.get('datahoraservidor', 0))
                        datahora = datetime.fromtimestamp(ts_servidor_ms / 1000.0) if ts_servidor_ms else None
                        lat = float(str(ponto.get('latitude', 0)).replace(',', '.'))
                        lon = float(str(ponto.get('longitude', 0)).replace(',', '.'))
                        velocidade = int(ponto.get('velocidade', 0))

                        pontos_para_inserir.append((
                            ponto.get('ordem'),
                            ponto.get('linha'),
                            velocidade,
                            datahora,
                            lat,
                            lon
                        ))
                        
                        if len(pontos_para_inserir) >= BATCH_SIZE:
                            inserir_lote(conn, pontos_para_inserir)
                            total_pontos_no_arquivo += len(pontos_para_inserir)
                            pontos_para_inserir.clear()

                    except (ValueError, TypeError, KeyError) as e:
                        print(f"  ⚠️  Aviso: pulando registro inválido. Erro: {e}. Dados: {ponto}")
                        continue
                
                if pontos_para_inserir:
                    inserir_lote(conn, pontos_para_inserir)
                    total_pontos_no_arquivo += len(pontos_para_inserir)

    except Exception as e:
        print(f"  ❌ Erro crítico ao processar {os.path.basename(filepath)}: {e}")
    
    print(f"  ✔️  {total_pontos_no_arquivo} pontos inseridos de {os.path.basename(filepath)}.")

def inserir_lote(conn, lote_dados):
    """Insere um lote de dados na tabela pontos_gps."""
    sql_com_geom = """
        INSERT INTO pontos_gps (ordem, linha, velocidade, datahora_servidor, latitude, longitude, geom)
        SELECT 
            v.ordem, v.linha, v.velocidade, v.datahora_servidor, v.latitude, v.longitude, 
            ST_SetSRID(ST_MakePoint(v.longitude, v.latitude), 4326)
        FROM (VALUES %s) AS v(ordem, linha, velocidade, datahora_servidor, latitude, longitude)
    """
    
    cursor = conn.cursor()
    try:
        execute_values(cursor, sql_com_geom, lote_dados, template=None, page_size=len(lote_dados))
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"❌ Erro ao inserir lote: {e}")
    finally:
        cursor.close()

def main():
    """Função principal para orquestrar o processo de carregamento."""
    conn = conectar_db()
    if not conn:
        return

    try:
        # Passo 1: Configurar a tabela e os índices
        setup_database(conn)

        # Passo 2: Iniciar o carregamento dos dados
        arquivos_zip = [f for f in os.listdir(DADOS_DIR) if f.endswith('.zip')]
        if not arquivos_zip:
            print(f"Nenhum arquivo .zip encontrado no diretório '{DADOS_DIR}'.")
            return

        print(f"\nEncontrados {len(arquivos_zip)} arquivos para processar.")

        for nome_arquivo in sorted(arquivos_zip):
            caminho_completo = os.path.join(DADOS_DIR, nome_arquivo)
            processar_arquivo_zip(caminho_completo, conn)

    except Exception as e:
        print(f"\nOcorreu um erro geral e o script foi interrompido: {e}")
    finally:
        # Garante que a conexão com o banco seja sempre fechada
        if conn:
            conn.close()
            print("\n🔌 Conexão com o banco de dados fechada.")

    print("\n🎉 Processo concluído!")

if __name__ == '__main__':
    main()
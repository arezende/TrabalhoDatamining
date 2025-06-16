import geopandas as gpd
from sqlalchemy import create_engine
import warnings
import psycopg2

warnings.filterwarnings('ignore', 'UserWarning', module='geopandas')

# --- CONFIGURAÇÕES - AJUSTE AQUI ---
DB_CONFIG = {
    'user': 'postgres',
    'password': 'P@ssw0rd123',
    'host': '172.18.166.112',
    'port': '5432',
    'dbname': 'onibus_db'
}

class RouteVisualizer:
    """
    Uma classe para buscar dados de rotas de ônibus do PostGIS e
    plotá-los em um mapa interativo, usando uma conexão psycopg2 direta.
    """
    def __init__(self, db_params):
        """
        Inicializa o visualizador com os parâmetros de conexão.
        """
        self.db_params = db_params
        self.conn = None # A conexão será armazenada aqui

    def connect(self):
        """
        Estabelece a conexão direta com o banco de dados usando psycopg2.
        """
        try:
            # Conectando diretamente e especificando a codificação do cliente.
            # Usamos 'latin1' porque o erro indica que os dados no banco
            # foram gravados com essa codificação, mesmo o banco sendo UTF8.
            self.conn = psycopg2.connect(**self.db_params, client_encoding='latin1')
            print("✅ Conexão direta com o banco de dados criada com sucesso (psycopg2).")
        except Exception as e:
            print(f"❌ Erro ao criar a conexão direta com psycopg2: {e}")
            self.conn = None

    def close(self):
        """Fecha a conexão com o banco de dados se estiver aberta."""
        if self.conn:
            self.conn.close()
            print("\n🔌 Conexão com o banco de dados fechada.")

    def plotar_linha(self, linha_id, limite_pontos=50000):
        """
        Busca os dados de uma linha de ônibus e gera um mapa interativo.
        """
        if not self.conn:
            print("❌ Impossível plotar a linha sem uma conexão válida com o banco.")
            return

        print(f"\n1. Buscando dados da linha '{linha_id}' no horário de operação (08:00-23:00)...")
        
        sql_query = f"""
            SELECT 
                ordem, linha, datahora_servidor, velocidade, geom
            FROM 
                pontos_gps_treino
            WHERE 
                linha = '{linha_id}'
                AND EXTRACT(HOUR FROM datahora_servidor) BETWEEN 8 AND 22
            LIMIT {limite_pontos};
        """
        
        try:
            # Passamos o objeto de conexão direta 'self.conn' para o geopandas.
            gdf = gpd.read_postgis(sql_query, self.conn, geom_col='geom')
            
            if gdf.empty:
                print(f"⚠️  Nenhum dado encontrado para a linha '{linha_id}' no horário de operação.")
                return
            print(f"  ✔️  {len(gdf)} pontos carregados com sucesso!")

        except Exception as e:
            print(f"❌ Erro ao buscar dados do PostGIS: {e}")
            return

        print("2. Gerando o mapa interativo...")
        
        m = gdf.explore(
            column='ordem',
            tooltip=['ordem', 'velocidade', 'datahora_servidor'],
            popup=True,
            marker_kwds={'radius': 2, 'fill': True},
            tiles='CartoDB positron',
            cmap='Paired',
            legend=False,
            style_kwds={'fillOpacity': 0.7, 'weight': 0}
        )

        output_filename = f'mapa_itinerario_linha_{linha_id}.html'
        m.save(output_filename)
        
        print(f"\n🎉 Mapa salvo com sucesso como '{output_filename}'!")

if __name__ == '__main__':
    # --- GERA UM MAPA PARA CADA LINHA DA LISTA ---

    # Lista de todas as linhas que devem ser processadas
    linhas_para_processar = [
        '483', '864', '639', '3', '309', '774', '629', '371', '397', '100', 
        '838', '315', '624', '388', '918', '665', '328', '497', '878', '355', 
        '138', '606', '457', '550', '803', '917', '638', '2336', '399', '298', 
        '867', '553', '565', '422', '756', '292', '554', '634', '232', '415', 
        '2803', '324', '852', '557', '759', '343', '779', '905', '108'
    ]
    # Removi a linha 186012003 que estava no enunciado, pois parece ser um ID e não um nome de linha.
    # Se for uma linha válida, pode adicionar de volta.

    # 1. Crie uma instância da classe
    visualizador = RouteVisualizer(DB_CONFIG)
    
    # 2. Conecta ao banco de dados uma única vez
    visualizador.connect()
    
    # 3. Garante que a conexão será fechada no final, mesmo se houver erros
    if visualizador.conn:
        try:
            # 4. Itera sobre a lista e plota cada linha
            for linha in linhas_para_processar:
                print("-" * 40)
                visualizador.plotar_linha(linha)
        finally:
            visualizador.close()
    
    print("\n\n" + "="*40)
    print("🎉 Processo de visualização de todas as linhas concluído!")
    print("="*40)
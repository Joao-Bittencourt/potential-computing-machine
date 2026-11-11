import os
import pendulum
import duckdb
from sklearn.ensemble import RandomForestClassifier
from airflow.decorators import dag, task

DB_PATH = "/opt/airflow/data/olist_dataset/olist_analytics.duckdb"
CSV_DIR = "/opt/airflow/data/olist_dataset"

@dag(
    dag_id="olist_duckdb_predictive_pipeline",
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    schedule="@once",
    catchup=False,
    tags=["olist", "duckdb", "ecommerce", "ml"]
)
def olist_pipeline():

    @task
    def ingest_csv_to_duckdb():
        con = duckdb.connect(DB_PATH)
        
        # Mapeamento do nome do arquivo para o nome da tabela
datasets = {
            "olist_customers_dataset.csv": "dim_customers",
            "olist_geolocation_dataset.csv": "dim_geolocation",
            "olist_order_items_dataset.csv": "fact_order_items",
            "olist_order_payments_dataset.csv": "fact_order_payments",
            "olist_order_reviews_dataset.csv": "fact_order_reviews",
            "olist_orders_dataset.csv": "fact_orders",
            "olist_products_dataset.csv": "dim_products",
            "olist_sellers_dataset.csv": "dim_sellers",
            "product_category_name_translate.csv": "dim_category_translation"
        }
        
        for filename, table_name in datasets.items():
            file_path = os.path.join(CSV_DIR, filename)
            
            if os.path.exists(file_path):
                print(f"Ingerindo {filename} na tabela {table_name}...")
                
                con.execute(f"""
                    CREATE OR REPLACE TABLE {table_name} AS 
                    SELECT * FROM read_csv_auto('{file_path}')
                """)
            else:
                print(f"Arquivo {filename} não encontrado em {CSV_DIR}.")
                
        con.close()
        print("Ingestão dos dados concluída com sucesso!")

    @task
    def build_features_and_predictive_model():
        con = duckdb.connect(DB_PATH)
        
        # consulta unindo pedidos, itens, produtos e clientes
        query = """
            SELECT 
                o.order_id,
                c.customer_state,
                p.product_category_name,
                oi.price,
                oi.freight_value,
                -- 1 se a entrega ocorreu depois da data estimada, 0 caso contrário
                CASE 
                    WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date THEN 1 
                    ELSE 0 
                END AS is_late
            FROM fact_orders o
            JOIN dim_customers c ON o.customer_id = c.customer_id
            JOIN fact_order_items oi ON o.order_id = oi.order_id
            JOIN dim_products p ON oi.product_id = p.product_id
            WHERE o.order_status = 'delivered'
              AND o.order_delivered_customer_date IS NOT NULL
              AND o.order_estimated_delivery_date IS NOT NULL
        """
        
        df = con.execute(query).fetchdf()
        con.close()
        
        if df.empty:
            print("Dataset vazio")
            return

        print(f"Dataset carregado. Total de linhas: {len(df)}")
        
        # Tratamento simples de valores nulos e categóricos para o modelo
        df = df.dropna()
        df['product_category_name'] = df['product_category_name'].astype('category').cat.codes
        df['customer_state'] = df['customer_state'].astype('category').cat.codes
        
        X = df[['customer_state', 'product_category_name', 'price', 'freight_value']]
        y = df['is_late']
        
        # modelo preditivo (RandomForest para prever risco de atraso)
        model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10)
        model.fit(X, y)
        
        score = model.score(X, y)
        print(f"Modelo treinado! Acurácia no dataset de treino: {score:.2f}")

        df['predicted_is_late'] = model.predict(X)
        df['prediction_probability'] = model.predict_proba(X)[:, 1] # Probabilidade de atraso
        
        
        df_results = df[[
            'order_id', 
            'customer_state', 
            'product_category_name', 
            'price', 
            'freight_value', 
            'is_late', 
            'predicted_is_late', 
            'prediction_probability'
        ]]
        
        con.execute("CREATE OR REPLACE TABLE pred_orders_late AS SELECT * FROM df_results")
        
        print("Previsões salvas")
        
        resumo = con.execute("SELECT predicted_is_late, COUNT(*) as total FROM pred_orders_late GROUP BY predicted_is_late").fetchall()
        print(f"Resumo das predições: {resumo}")
        
        con.close()

    
    ingest_csv_to_duckdb() >> build_features_and_predictive_model()

dag_instance = olist_pipeline()
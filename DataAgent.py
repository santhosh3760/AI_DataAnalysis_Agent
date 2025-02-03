import streamlit as st
import clickhouse_connect
from openai import OpenAI
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ClickHouse connection details
CLICKHOUSE_CONFIG = {
    'host': 'yt8dfy191y.eastus2.azure.clickhouse.cloud',
    'user': 'default',
    'password': 'a2~8U2HJe.uxs',
    'secure': True
}

class DatabaseAgent:
    def __init__(self):
        try:
            self.client = clickhouse_connect.get_client(**CLICKHOUSE_CONFIG)
            test_result = self.client.query("SELECT 1").result_set[0][0]
            if test_result == 1:
                print("Database connection successful!")
        except Exception as e:
            st.error(f"Failed to connect to ClickHouse: {str(e)}")
            raise e
    
    def get_table_schema(self):
        try:
            schema_query = "DESCRIBE trade_data"
            schema_df = pd.DataFrame(self.client.query(schema_query).result_set)
            return schema_df
        except Exception as e:
            print(f"Error fetching schema: {str(e)}")
            return None

    def get_table_preview(self, num_rows: int):
        try:
            preview_query = f"SELECT * FROM trade_data LIMIT {num_rows}"
            preview_df = self.client.query_df(preview_query)
            return preview_df
        except Exception as e:
            print(f"Error fetching table preview: {str(e)}")
            return None

    def generate_sql_query(self, user_query: str, api_key: str) -> str:
        if not api_key:
            raise ValueError("OpenAI API key is required!")
        
        client = OpenAI(api_key=api_key)
        schema_df = self.get_table_schema()
        schema_text = schema_df.to_string() if schema_df is not None else "Table: trade_data"
        
        prompt = f"""
        Given the following database schema:
        {schema_text}
        
        Convert this natural language query into a SQL query:
        "{user_query}"
        
        IMPORTANT: The table name is 'trade_data'. Always use this exact table name.
        
        Return only the SQL query, nothing else.
        Make sure the query is compatible with ClickHouse SQL syntax.
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a SQL expert. Generate only SQL queries without any explanation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")
    
    def execute_query(self, sql_query: str) -> pd.DataFrame:
        try:
            result = self.client.query_df(sql_query)
            return result
        except Exception as e:
            raise Exception(f"Query execution failed: {str(e)}")
    
    def close_connection(self):
        self.client.close()

def generate_visualizations(data: pd.DataFrame, user_query: str, api_key: str):
    if data.empty:
        return
    
    client = OpenAI(api_key=api_key)
    sample_data = data
    
    prompt = f"""
    create insightful visualizations for the below sample data:
    {sample_data}
    
    Based on the user query: "{user_query}", generate the most  insightful visualizations.
    Provide a Python code snippet using Matplotlib and Seaborn to generate these plots.
    Only return the code, no explanations.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a data visualization expert. Generate only Python code snippets."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        
        raw_code = response.choices[0].message.content.strip()
        
        # Remove markdown formatting if present
        if raw_code.startswith("```python"):
            raw_code = raw_code[9:]  
        if raw_code.endswith("```"):
            raw_code = raw_code[:-3]  

        # Display the generated code for reference
        st.text_area("Generated Visualization Code", raw_code, height=200)

        # Execute the cleaned code and capture the plots
        exec(raw_code, globals())

        # **Ensure plots are displayed in Streamlit**
        st.pyplot(plt)
        
    except Exception as e:
        st.error(f"Visualization generation failed: {str(e)}")

def main():
    st.title("Trade Data Analysis Agent")
    
    api_key = st.sidebar.text_input("Enter your OpenAI API key:", type="password")
    if not api_key:
        st.warning("Please enter your OpenAI API key in the sidebar to proceed.")
        st.stop()
    
    if 'openai_api_key' not in st.session_state:
        st.session_state.openai_api_key = api_key
    
    if 'agent' not in st.session_state:
        st.session_state.agent = DatabaseAgent()
    
    # Display the table schema in the sidebar
    st.sidebar.header("Table Schema")
    schema_df = st.session_state.agent.get_table_schema()
    if schema_df is not None:
        st.sidebar.dataframe(schema_df)
    else:
        st.sidebar.error("Failed to fetch table schema.")
    
    # Add a number input for selecting the number of rows to preview
    num_rows = st.sidebar.number_input(
        "Select number of rows to preview:",
        min_value=1,
        max_value=1000,
        value=10,  # Default value
        step=1
    )
    
    # Fetch and display the table preview based on the selected number of rows
    if num_rows > 0:
        st.sidebar.header(f"Preview of First {num_rows} Rows")
        preview_df = st.session_state.agent.get_table_preview(num_rows)
        if preview_df is not None:
            st.sidebar.dataframe(preview_df)
        else:
            st.sidebar.error("Failed to fetch table preview.")
    
    st.write("Ask questions about your trade data in natural language!")
    
    user_query = st.text_input("Enter your question:", 
                              placeholder="e.g., Show me all trades for AAPL in the last month")
    
    if user_query:
        try:
            with st.spinner("Generating SQL query..."):
                sql_query = st.session_state.agent.generate_sql_query(user_query, api_key)
                st.code(sql_query, language="sql")
            
            with st.spinner("Executing query..."):
                results = st.session_state.agent.execute_query(sql_query)
                
                if not results.empty:
                    st.dataframe(results)
                    csv = results.to_csv(index=False)
                    st.download_button(
                        label="Download results as CSV",
                        data=csv,
                        file_name="query_results.csv",
                        mime="text/csv"
                    )
                    
                    generate_visualizations(results, user_query, api_key)
                else:
                    st.info("No results found for your query.")
                    
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
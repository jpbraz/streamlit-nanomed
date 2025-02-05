import pandas as pd
import streamlit as st
from prophet import Prophet
import pymannkendall as mk
from prophet.plot import add_changepoints_to_plot


@st.cache_data
def converter_arquivo_em_dataframe(uploaded_file):
    if uploaded_file.type.endswith('csv'):
        # ler o arquivo csv e converter para dataframe com verificado automatico do separador
        try:
            data = pd.read_csv(uploaded_file, sep=None, engine='python')
            if data.empty:
                st.error("Arquivo vazio ou com formato inválido.")
                return pd.DataFrame(data=None)
        except pd.errors.ParserError:
            st.error("Formato de arquivo inválido. Por favor, verifique seu arquivo CSV.")
            return pd.DataFrame(data=None)
        except Exception as e:
            st.error(f"Erro ao ler o arquivo: {str(e)}")
            return pd.DataFrame(data=None)
        return data
    elif uploaded_file.type.endswith('xlsx') or uploaded_file.type.endswith('xls') or uploaded_file.type.endswith('sheet'):
        # ler o arquivo excel e converter para dataframe
        try:
            data = pd.read_excel(uploaded_file)
            if data.empty:
                st.error("Arquivo vazio ou com formato inválido.")
                return pd.DataFrame(data=None)
        except pd.errors.ParserError:
            st.error("Por favor, verifique o formato do arquivo. Utilize arquivos CSV ou Excel.")
            return pd.DataFrame(data=None)
        except Exception as e:
            st.error(f"Erro ao ler o arquivo: {str(e)}")
            return pd.DataFrame(data=None)
        return data
    else:
        data = pd.DataFrame(data=None)
        st.error("Formato de arquivo inválido. Por favor, utilize arquivos CSV ou Excel.")
    return data


CHOICES_TYPE_PERIOD = {'Y': 'Ano', 'm': 'Mês', 'd': 'Dia', 'H': 'Hora'}


def format_func(option):
    return CHOICES_TYPE_PERIOD[option]

# Estilização
with open("styles.css") as css:
    st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)

# Container Principal
with st.container():
    st.title("Seja bem-vindo ao NanomedPred/UFRN!")
    st.markdown('''
        ##### NanomedPred é uma aplicação web desenvolvida em Python, utilizando ferramentas de código aberto, que permite a utilização de ferramentas de Machine Learning Prophet (FBProphet) e pyMannKendall em uma plataforma web, para análises de predição de dados tabulados em planilhas, dessa maneira não requer conhecimento prévio em programação pelo usuário. A aplicação web permite uma fácil inserção e manipulação dos dados, e fornece uma análise de muitos dados de maneira rápida eficiente.
    ''', unsafe_allow_html=True)
    st.write('Para utiliza-la, basta seguir os passos orientados na barra lateral') 
    st.divider()
    st.markdown('''
            O uso da aplicação é gratuita. Caso seja utilizado em trabalho científico, solicitamos que o seja referenciado:
            NERY, D. ALBUQUERQUE, A., BRAZ, J.P.A., BRAZ, J. K. F.S. **NanomedPred, 2022**. Disponível em: _https://nanomed-emcm.streamlit.app/_ Acesso em: dia, mês e ano.
            ''')

# Barra Lateral (sidebar)
# Colocar Logo na Barra Lateral, e disponibiliza-la centralizada
image = "logo_nanomedpred.png"
st.sidebar.markdown(
        """
        <h1 style="text-align: center; margin-top: 0;">
            NanomedPred
        </h1>
        """,
        unsafe_allow_html=True
    )
st.sidebar.image(image, width=200)
st.sidebar.caption(
    "Predição de dados epidemiológico usando o pacote Prophet do Facebook (Open Source) para Procedimento de previsão automática")

# Carregar arquivo
st.sidebar.header("Carregar Arquivo")
st.sidebar.caption("O arquivo deve conter duas colunas: uma para a data e outra para os dados a serem previstos.")

uploaded_file = st.sidebar.file_uploader("Escolha um arquivo", type=['.csv', '.xls', '.xlsx'], 
                                accept_multiple_files=False,
                                disabled=False)

# Carregar dados
if uploaded_file is not None:
    # converter arquivo em dataframe
    data = converter_arquivo_em_dataframe(uploaded_file)

    if data is not None:
        
        st.success('Arquivo Carregado')

        # variaveis para campos de atributos das predições
        qnt_max = int(len(data.index)*0.20)

        # fig, ax = plt.subplots()
        # data.hist(bins=8, column=coluna_dados, grid=False, figsize=(4, 4), color="#86bf91", zorder=2, rwidth=0.9, ax=ax, )
        # st.subheader('Histograma')
        # st.write(fig)

        st.sidebar.subheader("Defina os atributos para predição")

        # mapeando dados do usuário para cada atributo
        periodo_dados = st.sidebar.selectbox("Coluna de tempo (data)", options=list(data.select_dtypes(exclude=['int64', 'float64'])))
        coluna_dados = st.sidebar.selectbox("Coluna dos valores (dados)", options=list(data.select_dtypes(include=['int64', 'float64'])))
        tipo_periodo_predicao = st.sidebar.selectbox("Tipo de Periodo para Predição",
                                                     options=list(CHOICES_TYPE_PERIOD.keys()), format_func=format_func)
        periodo_predicao = st.sidebar.slider(label='Período para predição', min_value=0, max_value=qnt_max, value=1)

        st.info(
            f"Selecionada a coluna '{coluna_dados}' com predição por '{format_func(tipo_periodo_predicao)}' para '{periodo_predicao}' períodos")

        if st.checkbox('Mostrar primeiros dados da planilha', ):
            st.subheader('Head dos dados ')
            st.write(data.head(n=6))

        if periodo_dados and coluna_dados:
            # inserindo um botão na tela
            btn_predict = st.sidebar.button("Realizar Predição", help="Predição usando o Prophet")
            btn_pymannkendall = st.sidebar.button("Análise de Tendências",
                                                  help="Análise de tendência usando o pymannkendall")

            # verifica se o botão foi acionado
            if btn_predict:
                data[periodo_dados] = pd.to_datetime(data[periodo_dados])
                data[periodo_dados] = data[periodo_dados].dt.strftime('%Y-%m-%d %H:%M:%S')
                # print(data[periodo_dados].to_list)

                # Prophet
                #df2 = pd.DataFrame.from_records(data)
                df = data.copy()
                df.rename(columns={periodo_dados: 'ds', coluna_dados: 'y'}, inplace=True)
                m = Prophet()
                m.fit(df)
                future = m.make_future_dataframe(periods=periodo_predicao, freq=tipo_periodo_predicao, include_history=True)
                # future.tail()
                forecast = m.predict(future)
                # forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
                fig1 = m.plot(forecast, xlabel=format_func(tipo_periodo_predicao), ylabel='Frequência')
                a = add_changepoints_to_plot(fig1.gca(), m, forecast)

                fig2 = m.plot_components(forecast)

                # st.subheader('Dados')
                # st.write(forecast)
                st.subheader('Gráfico com predição')
                st.write(fig1)
                st.write(fig2)

            if btn_pymannkendall:
                df_mannkendall = pd.DataFrame(data)
                dados = df_mannkendall[coluna_dados].to_list()

                result = mk.original_test(dados)

                st.subheader("Resultado do pyMannKendall")
                if result.trend == 'no trend':
                    st.error(result)
                else:
                    st.success(result)
                    st.write("Parabens! Seu dados tem uma dendência de: ", result.trend)
                    st.balloons()

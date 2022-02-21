import pandas as pd
import streamlit as st
from prophet import Prophet
import pymannkendall as mk


def converter_arquivo_em_dataframe(uploaded_file):
    if uploaded_file.type.endswith('csv'):
        data = pd.read_csv(uploaded_file, sep=";")
    elif uploaded_file.type.endswith('excel'):
        data = pd.read_excel(uploaded_file)
    else:
        data = pd.DataFrame(data=None)
    return data


CHOICES_TYPE_PERIOD = {'Y': 'Anual', 'm': 'Mensal', 'd': 'Diário'}


def format_func(option):
    return CHOICES_TYPE_PERIOD[option]


st.sidebar.title("Projeto NanoMed/UFRN")
st.sidebar.caption(
    "Predição de dados epidemiológico usando o pacote Prophet do Facebook (Open Source) para Procedimento de previsão automática")

uploaded_file = st.sidebar.file_uploader("Escolha um arquivo", type=['.csv', '.xls'], accept_multiple_files=False,
                                 disabled=False)

if uploaded_file is not None:
    print(uploaded_file.type)
    data = converter_arquivo_em_dataframe(uploaded_file)

    if data is not None:


        # fig, ax = plt.subplots()
        # data.hist(bins=8, column=coluna_dados, grid=False, figsize=(4, 4), color="#86bf91", zorder=2, rwidth=0.9, ax=ax, )
        # st.subheader('Histograma')
        # st.write(fig)

        st.sidebar.subheader("Defina os atributos para predição")

        # mapeando dados do usuário para cada atributo
        periodo_dados = st.sidebar.selectbox("Coluna da Data", options=list(data.select_dtypes(exclude='int64')))
        coluna_dados = st.sidebar.selectbox("Coluna de Dados", options=list(data.select_dtypes(include='int64')))
        tipo_periodo_predicao = st.sidebar.selectbox("Tipo de Periodo para Predição",
                                                     options=list(CHOICES_TYPE_PERIOD.keys()), format_func=format_func)
        periodo_predicao = st.sidebar.slider(label='Período para predição', min_value=1, max_value=20, value=5)
        st.info(
            f"Selecionada a coluna '{coluna_dados}' com predição '{format_func(tipo_periodo_predicao)}' para '{periodo_predicao}' períodos")

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
                # Prophet
                df = pd.DataFrame.from_records(data)
                df.rename(columns={periodo_dados: 'ds', coluna_dados: 'y'}, inplace=True)
                m = Prophet()
                m.fit(df)
                future = m.make_future_dataframe(periods=periodo_predicao, freq=tipo_periodo_predicao, include_history=True)
                # future.tail()
                futuro_predito = forecast = m.predict(future)
                # forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
                fig1 = m.plot(forecast, xlabel='Ano', ylabel='Casos')
                fig2 = m.plot_components(forecast)

                # st.subheader('Dados')
                # st.write(forecast)
                st.subheader('Gráfico com predição')
                st.write(fig1)
                st.write(fig2)

            if btn_pymannkendall:
                df_mannkendall = pd.DataFrame(data)
                dados = df_mannkendall[coluna_dados].to_list()

                print(dados)
                result = mk.original_test(dados)

                st.subheader("Resultado do pyMannKendall")
                st.success(result)
                st.balloons()

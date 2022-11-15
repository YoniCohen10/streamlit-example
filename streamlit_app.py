import math

import streamlit as st
import pandas as pd

###################################
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid.shared import JsCode

###################################

from functionforDownloadButtons import download_button
import dateutil.parser as parser

###################################
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import precision_score, confusion_matrix, fbeta_score
from sklearn.metrics import recall_score
import numpy as np
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import (precision_recall_curve,
                             PrecisionRecallDisplay)
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import pickle
from category_encoders import TargetEncoder
from io import BytesIO
import plotly.figure_factory as ff

pd.options.plotting.backend = "plotly"


def _max_width_():
    max_width_str = f"max-width: 1800px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )


# st.set_page_config(layout="wide")

st.set_page_config(page_icon="🧠", page_title="DSandbox", layout="wide", )

hide_menu = """
<style>
footer:before{
    content: 'Powered by OZ';
    display:block;
    position:relative;
    color:darkgrey;
    font-size:30px;
}
</style>
"""
st.markdown(hide_menu, unsafe_allow_html=True)

# st.image("https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/240/apple/285/balloon_1f388.png", width=100)

col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns((1, 1, 1, 1, 1, 1, 1, 1, 1))
col5.image(
    "https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/120/apple/325/person-running_1f3c3.png",
    width=100,
)

col1, col2, col3, col4, col5, col6, col7 = st.columns((1, 1, 1, 1, 1, 1, 1))
col4.title("DSandbox")

# st.caption(
#     "PRD : TBC | Streamlit Ag-Grid from Pablo Fonseca: https://pypi.org/project/streamlit-aggrid/"
# )


# ModelType = st.radio(
#     "Choose your model",
#     ["Flair", "DistilBERT (Default)"],
#     help="At present, you can choose between 2 models (Flair or DistilBERT) to embed your text. More to come!",
# )

# with st.expander("ToDo's", expanded=False):
#     st.markdown(
#         """
# -   Add pandas.json_normalize() - https://streamlit.slack.com/archives/D02CQ5Z5GHG/p1633102204005500
# -   **Remove 200 MB limit and test with larger CSVs**. Currently, the content is embedded in base64 format, so we may end up with a large HTML file for the browser to render
# -   **Add an encoding selector** (to cater for a wider array of encoding types)
# -   **Expand accepted file types** (currently only .csv can be imported. Could expand to .xlsx, .txt & more)
# -   Add the ability to convert to pivot → filter → export wrangled output (Pablo is due to change AgGrid to allow export of pivoted/grouped data)
# 	    """
#     )
# 
#     st.text("")

c29, c30, c31 = st.columns([1, 6, 1])

with c30:
    uploaded_file = st.file_uploader(
        "",
        key="1",
        help="To activate 'wide mode', go to the hamburger menu > Settings > turn on 'wide mode'",
    )

    if uploaded_file is not None:
        file_container = st.expander("Check your uploaded .csv")
        shows = pd.read_csv(uploaded_file)
        uploaded_file.seek(0)
        file_container.write(shows)

    else:
        st.info(
            f"""
                👆 Upload a .csv file first. Sample to try: [biostats.csv](https://people.sc.fsu.edu/~jburkardt/data/csv/biostats.csv)
                """
        )

        st.stop()

from st_aggrid import GridUpdateMode, DataReturnMode

gb = GridOptionsBuilder.from_dataframe(shows)
# enables pivoting on all columns, however i'd need to change ag grid to allow export of pivoted/grouped data, however it select/filters groups
gb.configure_default_column(enablePivot=True, enableValue=True, enableRowGroup=True)
gb.configure_selection(selection_mode="multiple", use_checkbox=True)
gb.configure_side_bar()  # side_bar is clearly a typo :) should by sidebar
gridOptions = gb.build()

st.success(
    f"""
        💡 Tip! Hold the shift key when selecting rows to select multiple rows at once!
        """
)

response = AgGrid(
    shows,
    gridOptions=gridOptions,
    enable_enterprise_modules=True,
    update_mode=GridUpdateMode.MODEL_CHANGED,
    data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
    fit_columns_on_grid_load=False,
)

df = pd.DataFrame(response["selected_rows"])

st.subheader("Filtered data will appear below 👇 ")
st.text("")

st.table(df)

st.text("")

c29, c30, c31 = st.columns([1, 1, 2])

with c29:
    CSVButton = download_button(
        df,
        "File.csv",
        "Download to CSV",
    )

with c30:
    TXTButton = download_button(
        df,
        "File.csv",
        "Download to TXT",
    )

ModelType = st.radio(
    "Choose your model",
    ["Classification (Default)", "Regression"],
    help="You need to choose the type of prediction you want to make about your target. More to come!",
)

target_feature = st.radio(
    "Choose your target columns",
    shows.columns.tolist(),
    help="You need to choose the column you want the model to predict its value",
)
if 'legit' not in st.session_state:
    st.session_state.legit = True

threshold = 2
label_size = len(shows[target_feature].value_counts())

if ModelType == 'Classification (Default)' and label_size > threshold:
    st.error(
        f"""
            💡 Note - You are trying to make a classification task with more then {threshold}, 
            currently your target label has {label_size} unique values, 
            currently supports only binary classification tasks.
            """
    )
    st.session_state.legit = False

if ModelType == 'Classification (Default)' and label_size == 2:
    st.session_state.legit = True
    st.success(
        f"""
            ✅  Your label column has exactly 2 values.
            """
    )

targrt_col_type = shows[target_feature].dtype
if ModelType == 'Regression' and targrt_col_type not in ['int64', 'float64', 'int', 'float']:
    st.error(
        f"""
                💡 Note - You are trying to make a regression task with feature type {targrt_col_type},
                regression tasks tries to predict continuous value i.e. - ints or floats 
                """
    )
    st.session_state.legit = False
else:
    if ModelType == 'Regression':
        st.session_state.legit = True
        st.success(
            f"""
                    ✅  Your label column has continuous values.
                    """
        )

random_or_date = st.radio(
    "Choose your split strategy",
    ['Random', 'By date'],
    help="You need to choose your split strategy, i.e. - the way you want to split your data to train and test sets.",
)

split_prop = ''
date_feature = ''
split_date = ''

if random_or_date == 'Random':
    cols = st.columns(1)
    split_prop = cols[0].slider("Train/test size:", 10, 100, 5)

else:
    date_feature = st.radio(
        "Choose your date columns",
        shows.columns.tolist(),
        help="You need to choose the column that represents the date of the sample"
    )
    cols = st.columns(1)
    try:
        st.session_state.legit = True
        split_date = cols[0].slider("Date split:", parser.parse(min(shows[date_feature]))
                                    , parser.parse(max(shows[date_feature])))
    except:
        st.session_state.legit = False
        st.error(
            f"""
                    ❌  The date format is unknown!
                    """
        )

col_to_drop = st.multiselect(
    '"Choose columns you want to drop from the table before training',
    shows.columns.tolist(),
    help="99.99% of the times features like date, ids, features with extremely high ratio of NaNs.")

st.write('You selected:', col_to_drop)

if ModelType == 'Classification (Default)':
    class_threshold = st.slider("Enter classification threshold:", min_value=0.01, max_value=0.99, value=0.5,
                                key='class_threshold')


def train_model(data, modelType, target_feature, random_or_date, split_prop, date_feature, split_date, col_to_drop):
    if date_feature in col_to_drop:
        col_to_drop.remove(date_feature)

    data = data.drop(col_to_drop, axis=1)
    label = target_feature

    if (modelType == 'Classification (Default)') and (target_feature not in data._get_numeric_data().columns):
        unique_values_of_target = data[target_feature].unique()
        data[target_feature] = data[target_feature].apply(lambda x: 1 if x == unique_values_of_target[0] else 0)

    X_train, X_test, y_train, y_test = '', '', '', ''

    if random_or_date == 'Random':
        if date_feature != '':
            data = data.drop(date_feature, axis=1)
        X_train, X_test, y_train, y_test = train_test_split(
            data.drop(label, axis=1), data[label], test_size=100 - split_prop, random_state=42)

    if random_or_date == 'By date':
        data[date_feature] = data[date_feature].apply(lambda x: parser.parse(x))

        train_set = data[data[date_feature] < split_date]
        test_set = data[data[date_feature] >= split_date]

        train_set = train_set.drop(date_feature, axis=1)
        test_set = test_set.drop(date_feature, axis=1)

        X_train = train_set.drop(label, axis=1).reset_index(drop=True)
        y_train = train_set[label].reset_index(drop=True)

        X_test = test_set.drop(label, axis=1).reset_index(drop=True)
        y_test = test_set[label].reset_index(drop=True)

    all_cols = X_train.columns
    num_cols = X_train._get_numeric_data().columns
    cat_cols = list(set(all_cols) - set(num_cols))

    X_train[target_feature] = y_train
    X_train.dropna(subset=[target_feature], inplace=True)
    X_train.reset_index(inplace=True, drop=True)
    y_train = X_train[target_feature]

    X_test[target_feature] = y_test
    X_test.dropna(subset=[target_feature], inplace=True)
    X_test.reset_index(inplace=True, drop=True)
    y_test = X_test[target_feature]

    if modelType == 'Classification (Default)':
        for col in cat_cols:
            encoder = TargetEncoder()
            X_train[f'{col}_encoded'] = encoder.fit_transform(X_train[col], X_train[target_feature])
            X_train.drop(col, axis=1, inplace=True)

            X_test[f'{col}_encoded'] = encoder.transform(X_test[col])
            X_test.drop(col, axis=1, inplace=True)
    else:
        X_train.drop(cat_cols, axis=1, inplace=True)
        X_test.drop(cat_cols, axis=1, inplace=True)

    X_train.drop(target_feature, axis=1, inplace=True)
    X_test.drop(target_feature, axis=1, inplace=True)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test)

    num_round = min(int(math.sqrt(data.shape[1])), 8)
    depth = min(int(math.sqrt(data.shape[0])), 300)

    if modelType == 'Classification (Default)':
        param = {'max_depth': depth, 'eta': 1, 'objective': 'binary:logistic', 'n_jobs': -1, 'verbosity': 0,
                 'nthread': 48,
                 'colsample_bytree': 1, 'subsample': 1}
    else:
        param = {'max_depth': depth, 'eta': 1, 'objective': 'reg:squarederror', 'n_jobs': -1, 'verbosity': 0,
                 'nthread': 48,
                 'colsample_bytree': 1, 'subsample': 1}

    bst = xgb.train(param, dtrain, num_round)
    pereds = bst.predict(dtest)
    st.success('''Training complete!''')
    return bst, pereds, X_train, X_test, y_train, y_test, param


if shows.drop(col_to_drop, axis=1).shape[1] < 2:
    st.session_state.legit = False
# else:
#     legit = True

if st.session_state.legit and col_to_drop.count(target_feature) < 1:
    st.success('✅ Looks like all the training definitions are great! press Train model and start training!')

if not st.session_state.legit or not (col_to_drop.count(target_feature) < 1):
    st.error(
        '❌ Looks like something with the training definition is wrong, please double check you training definitions')

train_over = False
col1, col2, col3, col4, col5, col6, col7 = st.columns((1, 1, 1, 1, 1, 1, 1))
if col4.button('Train model!') and st.session_state.legit and col_to_drop.count(target_feature) < 1:
    try:
        # st.success(f""" 🏃  Everything looks great! Start Training!""")
        with st.spinner('Wait for it...'):
            bst, pereds, X_train, X_test, y_train, y_test, param = train_model(shows, ModelType, target_feature,
                                                                               random_or_date,
                                                                               split_prop, date_feature, split_date,
                                                                               col_to_drop)
        train_over = True
    except Exception as e:
        train_over = False
        st.error(str(e))
        st.error(
            '❌ It looks like some of the columns you have provided for training are not sutiable for training. please remove them befre training')

    images_to_save = []
    if ModelType == 'Classification (Default)' and train_over:
        st.balloons()
        pereds_label = np.where(pereds > class_threshold, 1, 0)
        cf_matrix = confusion_matrix(y_test, pereds_label)

        tn, fp, fn, tp = cf_matrix.ravel()
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        precision_l, recall_l, _ = precision_recall_curve(y_test, pereds)

        disp = PrecisionRecallDisplay(precision=precision_l, recall=recall_l)

        st.header('Cumulative KPIs')
        col1, col2, col3, col4 = st.columns((1, 1, 1, 1))

        col1.subheader("Precision")
        col1.write(str(precision))

        col2.subheader("Recall")
        col2.write(str(recall))

        col3.subheader('F1-Score')
        f1_score = fbeta_score(y_test, pereds_label, average='binary', beta=1)
        col3.write(str(f1_score))

        col4.subheader('F2-Score')
        f2_score = fbeta_score(y_test, pereds_label, average='binary', beta=2)
        col4.write(str(f2_score))

        st.header('Graphs')
        right, left = st.columns((1, 1))

        right.subheader("Precision recall curve")
        disp.plot()
        fig = plt.gcf()
        fig.set_size_inches(7, 5)
        images_to_save.append(fig)

        buf = BytesIO()
        fig.savefig(buf, format="png")
        right.image(buf)

        # right.pyplot(fig)
        plt.clf()

        left.subheader("Predictions histogram")
        plt.hist(pereds)
        fig1 = plt.gcf()
        fig1.set_size_inches(7, 5)
        plt.xlabel('Probs', fontsize=9)
        plt.ylabel('Amount', fontsize=9)

        images_to_save.append(fig1)

        buf = BytesIO()
        fig1.savefig(buf, format="png")
        left.image(buf)

        # plt.plot()
        # left.pyplot(fig1)

    elif ModelType == 'Regression' and train_over:
        st.balloons()
        st.subheader("MSE")
        st.write(str(mean_squared_error(y_test, pereds)))

        # st.subheader("Predictions histogram")
        # plt.hist(pereds)
        # plt.plot()
        # plt.xlabel(target_feature, fontsize=9)
        # plt.ylabel('Amount', fontsize=9)
        #
        # fig = plt.gcf()
        # fig.set_size_inches(7, 5)
        #
        # buf = BytesIO()
        # fig.savefig(buf, format="png")
        # st.image(buf)

        fig = ff.create_distplot([np.transpose(pereds)], ['Regressor predictions'], show_curve=False, colors=['red'],
                                 histnorm='')
        fig.update_layout(xaxis=dict(title=f'Count'), yaxis=dict(title=f'{target_feature}'),
                          title=f'{target_feature} prediction distribution',
                          height=800)

        # Plot!
        st.plotly_chart(fig, use_container_width=True)

        # images_to_save.append(fig)
        # st.pyplot(fig)
    if train_over:
        X_test['predictions'] = pereds
        X_test['label'] = y_test
        X_train['label'] = y_train
        model_to_save = bst
        model_parameters = param
        l = []

        st.header('Download experiment artifacts')
        c1, c2, c3 = st.columns((1, 1, 1))

        with c1:
            download_button(X_train,
                            "train_data.csv",
                            "⬇️ Train data")
        with c2:
            download_button(X_test,
                            "test_data.csv",
                            "⬇️ Test data")
        with c3:
            download_button(pickle.dumps(model_to_save),
                            "model.pkl",
                            "⬇️ Trained model")

    # from matplotlib.backends.backend_pdf import PdfPages
    #
    # pp = PdfPages('foo.pdf')
    # for im in images_to_save:
    #     pp.savefig(im)
    #
    # pp.close()
    # with c4:
    #     download_button(pp,
    #                     "images.pdf",
    #                     "⬇️ Images")


else:
    # st.balloons()
    if st.session_state.legit:
        st.success("Press Train model and start training!")

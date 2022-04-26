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
from sklearn.metrics import precision_score, confusion_matrix
from sklearn.metrics import recall_score
import numpy as np
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import (precision_recall_curve,
                             PrecisionRecallDisplay)
import matplotlib.pyplot as plt


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


st.set_page_config(page_icon="🧠", page_title="DSandbox")

# st.image("https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/240/apple/285/balloon_1f388.png", width=100)
st.image(
    "https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/120/apple/325/brain_1f9e0.png",
    width=100,
)

st.title("DSandbox")

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

legit = True

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
    legit = False
if ModelType == 'Classification (Default)' and label_size == 2:
    legit = True
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
    legit = False
else:
    legit = True
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
        legit = True
        split_date = cols[0].slider("Date split:", parser.parse(min(shows[date_feature]))
                                    , parser.parse(max(shows[date_feature])))
    except:
        legit = False
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


def train_model(data, modelType, target_feature, random_or_date, split_prop, date_feature, split_date, col_to_drop):
    if date_feature in col_to_drop: col_to_drop.remove(date_feature)
    data = data.drop(col_to_drop, axis=1)
    label = target_feature
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

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test)
    num_round = 100
    if modelType == 'Classification (Default)':
        param = {'max_depth': 3, 'eta': 1, 'objective': 'binary:logistic', 'n_jobs': -1, 'verbosity': 2, 'nthread': 48,
                 'colsample_bytree': 0.8, 'subsample': 0.8}
    else:
        param = {'max_depth': 3, 'eta': 1, 'objective': 'reg:squarederror', 'n_jobs': -1, 'verbosity': 2, 'nthread': 48,
                 'colsample_bytree': 0.8, 'subsample': 0.8}
    bst = xgb.train(param, dtrain, num_round)
    pereds = bst.predict(dtest)
    st.success('''Training complete!''')
    return bst, pereds, X_train, X_test, y_train, y_test


if shows.drop(col_to_drop, axis=1).shape[1] < 2:
    legit = False
else:
    legit = True

if legit and col_to_drop.count(target_feature) < 1:
    st.success('✅ Looks like all the training defenitions are gread! press Train model and start training!')

if not legit or not (col_to_drop.count(target_feature) < 1):
    st.error(
        '❌ Looks like something with the training definition is wrong, please double check you training definitions')

if st.button('Train model!') and legit and col_to_drop.count(target_feature) < 1:
    st.success(f""" 🏃  Everything looks great! Start Training!""")
    with st.spinner('Wait for it...'):
        bst, pereds, X_train, X_test, y_train, y_test = train_model(shows, ModelType, target_feature, random_or_date,
                                                                    split_prop, date_feature, split_date, col_to_drop)
    # st.balloons()
else:
    st.error("123")


left, right = st.columns(2)

if ModelType == 'Classification (Default)':
    form = left.form('show_results')
    class_threshold = form.slider("enter classification threshold:", min_value=0.01, max_value=0.99, value=0.5,
                                  key='class_threshold')
    st.write(f'{type(class_threshold)}')
    submit = form.form_submit_button("Refresh results")

    if submit:
        pereds_label = np.where(pereds > class_threshold, 1, 0)
        st.write("1")
        cf_matrix = confusion_matrix(y_test, pereds_label)

        tn, fp, fn, tp = cf_matrix.ravel()
        st.write("2")
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        # precision_l, recall_l, _ = precision_recall_curve(y_test, pereds)
        #
        # disp = PrecisionRecallDisplay(precision=precision_l, recall=recall_l)
        st.write("3")
        right.header("Precision")
        right.write(str(precision))

        # st.header("Recall")
        # st.write(str(recall))
        #
        # st.header("prcision recall curve")
        # fig, ax = plt.subplots()
        # st.pyplot(fig)
# else:
#     pass
st.write('end')

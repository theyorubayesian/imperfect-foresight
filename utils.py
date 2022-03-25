from pathlib import Path

import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay
)


def create_train_test_data(
    data: pd.DataFrame,
    output_dir: str = None, 
    cut_off: int = 1, 
    return_df: bool = True, 
    write_df: bool = False
) -> None:
    """
    
    """
    temp = data.copy()
    temp["event_time"] = pd.to_datetime(temp["event_time"])

    cut_off_date = max(temp["event_time"]) - pd.DateOffset(months=cut_off)
    train = temp.query("event_time < @cut_off_date")
    test = temp.query("event_time >= @cut_off_date")

    # Pick first interaction per user in test set
    test = test.groupby("user_id").first().reset_index()

    if write_df and output_dir != None:
        output_dir = Path(output_dir)
        train.to_csv(output_dir / "train.csv", index=False)
        test.to_csv(output_dir / "test.csv", index=False)
    
    if return_df:
        return train, test

    return


def calc_next_event(session_list: DataFrameGroupBy) -> pd.DataFrame:
    idxs = []
    events = []
    
    for session in session_list:
        session_id, session_data = session

        if len(session_data) == 1:
            idxs.extend(list(session_data.index))
            events.append("exit")
        else:
            idxs.extend(list(session_data.index))
            session_events = session_data["event_type"].iloc[1:].tolist() + ["exit"]
            events.extend(session_events)
    
    return pd.DataFrame(events, index=idxs, columns=["next_event"])


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    temp = df.copy()
    temp.dropna(subset=["user_session"], inplace=True)
    temp["event_time"] = pd.to_datetime(temp["event_time"])

    temp["month"] = temp["event_time"].dt.month
    temp["day"] = temp["event_time"].dt.day
    temp["hour"] = temp["event_time"].dt.hour

    temp["category_code__clean"] = (
        temp["category_code"]
        .fillna("missing")
        .apply(lambda x: x.split(".")[0])
    )
    return temp


def eval_model(clf, X_test, y_test, return_preds: bool = True, include_plots: bool = True, mode="eval"):
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = \
        precision_recall_fscore_support(y_test, y_pred, average="macro")
    
    results = {
        f"{mode}_accuracy": accuracy,
        f"{mode}_precision": precision,
        f"{mode}_recall": recall,
        f"{mode}_f1": f1,
    }

    if return_preds:
        results[f"{mode}_y_pred"] = y_pred

    if include_plots:
        cm_image = ConfusionMatrixDisplay.from_estimator(
            clf, X_test, y_test, display_labels=["Not Purchase", "Purchase"])
        pr_image = PrecisionRecallDisplay.from_estimator(clf, X_test, y_test)

        results[f"{mode}_confusion_matrix"] = cm_image
        results[f"{mode}_precision_recall_curve"] = pr_image

    return results

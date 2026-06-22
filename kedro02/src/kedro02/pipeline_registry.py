from kedro.pipeline import Pipeline
from kedro02.pipelines.pipeline import create_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    main_pipeline = create_pipeline()

    return {
        "__default__": main_pipeline,
        "car_price": main_pipeline,
    }

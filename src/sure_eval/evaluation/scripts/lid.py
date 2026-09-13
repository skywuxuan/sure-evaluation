"""LID configured script route descriptor and runner."""

from __future__ import annotations

from sure_eval.evaluation.scripts.contracts import (
    call_route_executor,
    contract_from_manifest,
    describe_from_contracts,
    find_pipeline_route,
    find_task_route,
    load_task_manifest,
    load_task_routes,
    write_route_run_outputs,
)


def describe_pipeline(*, metric: str = "accuracy", pipeline_id: str | None = None):
    manifest, manifest_path, routes_path, route = _select_route(
        metric=metric,
        pipeline_id=pipeline_id,
    )
    return describe_from_contracts(
        task="LID",
        pipeline_id=route["pipeline_id"],
        metric="accuracy",
        language="n/a",
        node_ids=tuple(route["nodes"]),
        contracts=(contract_from_manifest(manifest, route["input_contract"]),),
        task_config_path=manifest_path,
        route_config_path=routes_path,
        computation_node_ids=tuple(route["nodes"]),
        execution_metrics=("accuracy",),
        script_module=__name__,
        executor=str(route["executor"]),
    )


def run(
    ref_file: str,
    hyp_file: str,
    *,
    output_dir: str,
    pipeline_id: str | None = None,
):
    if not output_dir:
        raise ValueError("output_dir is required")
    _, _, _, route = _select_route(metric="accuracy", pipeline_id=pipeline_id)
    description = describe_pipeline(pipeline_id=route["pipeline_id"])
    if not ref_file or not hyp_file:
        raise ValueError("LID label evaluation requires ref_file and hyp_file")
    report = call_route_executor(route, ref_file=ref_file, hyp_file=hyp_file)
    return write_route_run_outputs(report=report, description=description, output_dir=output_dir)


def _select_route(*, metric: str, pipeline_id: str | None):
    if metric.lower() != "accuracy":
        raise ValueError(f"Unsupported LID metric: {metric}")
    manifest, manifest_path = load_task_manifest("lid")
    routes, routes_path = load_task_routes("lid")
    route = (
        find_pipeline_route(routes, pipeline_id=pipeline_id)
        if pipeline_id
        else find_task_route(routes, metric="accuracy")
    )
    return manifest, manifest_path, routes_path, route

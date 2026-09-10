"""Prove the Ray job runs MY branch's flow control, not the image-baked copy."""

import os

import py_inference_scheduler
from py_inference_scheduler import Scheduler
from py_inference_scheduler.core import flow_control
from py_inference_scheduler.framework.registry import _FLOW_CONTROLS

import integration.verl.verl_hook as hook

print("package  :", py_inference_scheduler.__file__)
print("flow_ctl :", flow_control.__file__)
print("hook     :", hook.__file__)
print("layout   :", hook._VERL_LAYOUT)
print("registered flow controls:", sorted(_FLOW_CONTROLS))
print("ROUTER_CONFIG_PATH:", os.environ.get("ROUTER_CONFIG_PATH"))

sched = Scheduler()
plugins = sched.get_flow_control_plugins()
print("plugins at construction:", [type(p).__name__ for p in plugins])
for p in plugins:
    print("   thresholds: kv=", p.kv_threshold, "waiting=", p.waiting_threshold)

assert "/opt/py-inference-scheduler" not in flow_control.__file__, (
    "job is importing the IMAGE-BAKED copy, not the shipped working_dir"
)
assert "simple_backpressure" in _FLOW_CONTROLS, "plugin not registered"
print("VERIFY OK")

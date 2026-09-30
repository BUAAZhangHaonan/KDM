"""Run the unchanged native chain with the separate A100 checkpoint-location registry."""
import native
import execution

execution.REGISTRY = "workflows/paper_core/host_registry_a100.json"

if __name__ == "__main__":
    raise SystemExit(native.main())

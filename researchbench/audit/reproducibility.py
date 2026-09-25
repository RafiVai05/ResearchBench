import sys
import platform
import datetime
import researchbench

def get_reproducibility_info() -> dict:
    return {
        "researchbench_version": researchbench.__version__,
        "python_version": sys.version.split(' ')[0],
        "os": platform.system(),
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
    }
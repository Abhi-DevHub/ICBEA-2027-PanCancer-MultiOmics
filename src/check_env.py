import os
import sys
import platform


def _try_import(name):
    try:
        mod = __import__(name)
        return True, mod
    except Exception as e:
        return False, e


def main():
    print("=== Environment Check ===")
    print("Python:", sys.version.replace("\n", " "))
    print("Platform:", platform.platform())
    print("CWD:", os.getcwd())
    print()

    for pkg in ["numpy", "pandas", "sklearn", "xgboost", "matplotlib", "seaborn"]:
        ok, mod = _try_import(pkg)
        if ok:
            ver = getattr(mod, "__version__", "?")
            print(f"OK  {pkg:10s}  {ver}")
        else:
            print(f"MISS {pkg:10s}  {mod}")

    print()

    ok_cp, cp = _try_import("cupy")
    ok_cudf, cudf = _try_import("cudf")
    ok_cuml, cuml = _try_import("cuml")

    if ok_cp:
        try:
            n_gpus = cp.cuda.runtime.getDeviceCount()
            dev = cp.cuda.Device(0)
            props = cp.cuda.runtime.getDeviceProperties(0)
            name = props.get("name", b"?")
            if isinstance(name, (bytes, bytearray)):
                name = name.decode("utf-8", errors="ignore")
            print(f"OK  cupy       GPUs: {n_gpus} | Device0: {name}")
        except Exception as e:
            print("WARN cupy       present but GPU query failed:", e)
    else:
        print("MISS cupy       (GPU arrays not available)")

    if ok_cudf:
        print("OK  cudf       ", getattr(cudf, "__version__", "?"))
    else:
        print("MISS cudf       (GPU DataFrame not available)")

    if ok_cuml:
        print("OK  cuml       ", getattr(cuml, "__version__", "?"))
    else:
        print("MISS cuml       (GPU ML models not available)")

    print()

    # Quick data presence check
    expected = [
        "Pan-cancer_mRNA.csv",
        "Pan-cancer_miRNA.csv",
        "Pan-cancer_CNV.csv",
        "Pan-cancer_Methy.csv",
        "Pan-cancer_label_num.csv",
    ]

    missing = [p for p in expected if not os.path.exists(p)]
    if missing:
        print("WARN Missing data files:")
        for p in missing:
            print(" -", p)
    else:
        print("OK  All expected data files present.")


if __name__ == "__main__":
    main()

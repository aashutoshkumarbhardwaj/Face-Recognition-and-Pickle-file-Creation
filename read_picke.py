import pickle
import os
from parameters import DLIB_FACE_ENCODING_PATH


def main():
    path = DLIB_FACE_ENCODING_PATH
    # helpful fallback: also accept common misspelling used earlier
    alt_path = path.replace('gril_', 'grill_') if 'gril_' in path else None

    if not os.path.exists(path):
        if alt_path and os.path.exists(alt_path):
            path = alt_path
        else:
            print(f"ERROR: encoding file not found: {DLIB_FACE_ENCODING_PATH}")
            print("Looked for:")
            print("  ", DLIB_FACE_ENCODING_PATH)
            if alt_path:
                print("  ", alt_path)
            print("Available pickles in data/:")
            for p in sorted([f for f in os.listdir('data') if f.endswith('.pkl')]):
                print('  ', p)
            raise SystemExit(1)

    with open(path, 'rb') as file:
        data = pickle.load(file)

    # Print a short summary rather than raw dump
    if isinstance(data, dict):
        enc = data.get('encodings') or []
        names = data.get('names') or []
        print(f"Loaded pickle: {path}")
        print(f"encodings: {len(enc)}  names: {len(names)}")
        print("sample names:", names[:10])
    else:
        print("Loaded object (non-dict):", type(data))
        try:
            print(data)
        except Exception:
            print("(could not print object)")


if __name__ == '__main__':
    main()
# chess-data


This repository holds the files needed to create the datasets needed for training our models. The debug dataset is just a smaller version of the main dataset for quickly testing new code on your local machine, whilst the main dataset is meant to be downloaded on the machine that will actually be used for training, whether that's a server, workstation, or compute cluster. The data is originally retrieved from [lichess](https://database.lichess.org/) Jan. 2021 to Mar. 2024, and is then binned into different elos. Each folder `1000`, `1100`, etc. corresponds to all games between `900-1000` and `1000-1100`, respectively. The main dataset contains roughly 3.2B games. Since each game is around 400 characters or so, this very roughly corresponds to ~1.3T tokens. These numbers were done back of hand, and could certainly be made more precise.

The data processing pipeline goes in the following order:

download -> uncompress -> split into chunks -> bin -> compress

download:
```
for y in [2021, 2022, 2023, 2024]:
    for m in range(1, 13):
        os.system(f'echo "wget https://database.lichess.org/standard/lichess_db_standard_rated_{y}-{m:02d}.pgn.zst"')
```

uncompress (note that this will require around 20TB and several hours):

```
for file in os.listdir('.'):
    os.system(f"zstd -d -T0 {file}")
```

split into chunks (this is done so that our zstd_process in chess-research is able to open enough file handlers:

```
for file in os.listdir('.'):
    if not file.endswith('zst'):
        os.system(f"split -d -n l/32 {file} {file}.")
```

Then you can run `elo_bin.py` to bin, and `zstd_compress_elo_bin.py` to compress.


The debug dataset can be created from the main one using the following code:

```
import os
import shutil

def copy_first_file(source_dir, target_dir):
    for root, dirs, files in os.walk(source_dir):
        if not files:
            continue  # Skip if there are no files in the current directory        

        # Get the relative path of the current directory with respect to the source_dir
        relative_path = os.path.relpath(root, source_dir)

        # Construct the corresponding target directory
        target_subdir = os.path.join(target_dir, relative_path)

        # Ensure the target subdirectory exists
        os.makedirs(target_subdir, exist_ok=True)

        # Copy the first file in the current directory to the target directory
        first_file = files[0]
        source_file_path = os.path.join(root, first_file)
        target_file_path = os.path.join(target_subdir, first_file)

        if '.git' in source_file_path:
            continue

        shutil.copy2(source_file_path, target_file_path)
        print(f"Copied {source_file_path} to {target_file_path}")

# Example usage
source_directory = "lichess_elo_binned"
target_directory = "lichess_elo_binned_debug"
copy_first_file(source_directory, target_directory)
```


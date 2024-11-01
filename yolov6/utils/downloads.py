from pathlib import Path
import torch
import urllib
import requests
import subprocess

def attempt_download_from_hub(repo_id, hf_token=None):
    # https://github.com/fcakyon/yolov5-pip/blob/main/yolov5/utils/downloads.py
    from huggingface_hub import hf_hub_download, list_repo_files
    from huggingface_hub.utils._errors import RepositoryNotFoundError
    from huggingface_hub.utils._validators import HFValidationError
    try:
        repo_files = list_repo_files(repo_id=repo_id, repo_type='model', token=hf_token)
        model_file = [f for f in repo_files if f.endswith('.pt')][0]
        file = hf_hub_download(
            repo_id=repo_id,
            filename=model_file,
            repo_type='model',
            token=hf_token,
        )
        return file
    except (RepositoryNotFoundError, HFValidationError):
        return None

def safe_download(file, url, url2=None, min_bytes=1E0, error_msg=''):
    import os
    # Attempts to download file from url or url2, checks and removes incomplete downloads < min_bytes

    file = Path(file)
    assert_msg = f"Downloaded file '{file}' does not exist or size is < min_bytes={min_bytes}"
    try:  # url1
        torch.hub.download_url_to_file(url, str(file), progress=True)  # pytorch download
        assert file.exists() and file.stat().st_size > min_bytes, assert_msg  # check
    except Exception as e:  # url2
        file.unlink(missing_ok=True)  # remove partial downloads
        os.system(f"curl -L '{url2 or url}' -o '{file}' --retry 3 -C -")  # curl download, retry and resume on fail
    finally:
        if not file.exists() or file.stat().st_size < min_bytes:  # check
            file.unlink(missing_ok=True)  # remove partial downloads
            raise Exception(error_msg or assert_msg)  # raise informative error

def attempt_download(file, repo='meituan/YOLOv6', release='0.4.0'):
    def github_assets(repository, version='latest'):
        response = requests.get(f'https://api.github.com/repos/{repository}/releases/tags/{version}').json()  # github api
        return response['tag_name'], [x['name'] for x in response['assets']]  # tag, assets

    file = Path(str(file).strip().replace("'", ''))
    if not file.exists():
        # URL specified
        name = Path(urllib.parse.unquote(str(file))).name  # decode '%2F' to '/' etc.
        if str(file).startswith(('http:/', 'https:/')):  # download
            url = str(file).replace(':/', '://')  # Pathlib turns :// -> :/
            file = name.split('?')[0]  # parse authentication https://url.com/file.txt?auth...
            if Path(file).is_file():
                return file
            else:
                safe_download(file=file, url=url, min_bytes=1E5)
            return file

        # GitHub assets
        assets = [
            'yolov6n.pt', 'yolov6s.pt', 'yolov6m.pt', 'yolov6l.pt',
            'yolov6n6.pt', 'yolov6s6.pt', 'yolov6m6.pt', 'yolov6l6.pt']
        try:
            tag, assets = github_assets(repo, release)
        except Exception:
            try:
                tag, assets = github_assets(repo)  # latest release
            except Exception:
                try:
                    tag = subprocess.check_output('git tag', shell=True, stderr=subprocess.STDOUT).decode().split()[-1]
                except Exception:
                    tag = release

        file.parent.mkdir(parents=True, exist_ok=True)  # make parent dir (if required)
        if name in assets:
            safe_download(
                file,
                url=f'https://github.com/{repo}/releases/download/{tag}/{name}',
                url2=f'https://storage.googleapis.com/{repo}/{tag}/{name}',  # backup url (optional)
                min_bytes=1E5,
                error_msg=f'{file} missing, try downloading from https://github.com/{repo}/releases/{tag}')
    
    return str(file)

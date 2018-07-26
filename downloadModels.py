from __future__ import print_function
import requests
import zipfile
import warnings
from sys import stdout
from os import makedirs
from os.path import dirname
from os.path import exists


class GoogleDriveDownloader:
    """
    Minimal class to download shared files from Google Drive.
    """

    CHUNK_SIZE = 32768
    DOWNLOAD_URL = "https://docs.google.com/uc?export=download"

    @staticmethod
    def download_file_from_google_drive(file_id, dest_path, overwrite=False, unzip=False):
        """
        Downloads a shared file from google drive into a given folder.
        Optionally unzips it.

        Parameters
        ----------
        file_id: str
            the file identifier.
            You can obtain it from the sherable link.
        dest_path: str
            the destination where to save the downloaded file.
            Must be a path (for example: './downloaded_file.txt')
        overwrite: bool
            optional, if True forces re-download and overwrite.
        unzip: bool
            optional, if True unzips a file.
            If the file is not a zip file, ignores it.

        Returns
        -------
        None
        """

        destination_directory = dirname(dest_path)
        if not exists(destination_directory):
            makedirs(destination_directory)

        if not exists(dest_path) or overwrite:

            session = requests.Session()

            print('Downloading {} into {}... '.format(file_id, dest_path), end='')
            stdout.flush()

            response = session.get(GoogleDriveDownloader.DOWNLOAD_URL, params={'id': file_id}, stream=True)

            token = GoogleDriveDownloader._get_confirm_token(response)
            if token:
                params = {'id': file_id, 'confirm': token}
                response = session.get(GoogleDriveDownloader.DOWNLOAD_URL, params=params, stream=True)

            GoogleDriveDownloader._save_response_content(response, dest_path)
            print('Done.')

            if unzip:
                try:
                    print('Unzipping...', end='')
                    stdout.flush()
                    with zipfile.ZipFile(dest_path, 'r') as z:
                        z.extractall(destination_directory)
                    print('Done.')
                except zipfile.BadZipfile:
                    warnings.warn('Ignoring `unzip` since "{}" does not look like a valid zip file'.format(file_id))

    @staticmethod
    def _get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    @staticmethod
    def _save_response_content(response, destination):
        with open(destination, "wb") as f:
            for chunk in response.iter_content(GoogleDriveDownloader.CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

if __name__ == "__main__":


    # https://drive.google.com/file/d/1kfSWY8AydXx9d3rFW2bjMyvBgXp0Nf01/view?usp=sharing
    glove_hdf5 = "1kfSWY8AydXx9d3rFW2bjMyvBgXp0Nf01"
    # https://drive.google.com/file/d/1AqyF8wcJfm6oNLyFLhLb-oESKRx3plvV/view?usp=sharing
    local_300_parikh_hdf5 = "1AqyF8wcJfm6oNLyFLhLb-oESKRx3plvV"
    # https://drive.google.com/file/d/1yq8pLCRKqVGQV0DXr_ZhBczIYJqs16pI/view?usp=sharing
    local_300_parikh_txt = "1yq8pLCRKqVGQV0DXr_ZhBczIYJqs16pI"
    # https://drive.google.com/file/d/1ZSKeS4nE8nYwMPA9q1ui6BMVJ4AaKAe1/view?usp=sharing
    snli_10_test_hdf5 = "1ZSKeS4nE8nYwMPA9q1ui6BMVJ4AaKAe1"
    # https://drive.google.com/file/d/1-FQQiCrxrH-0-F-CwI8bu_UZ_96ro81i/view?usp=sharing
    snli_10_train_hdf5 = "1-FQQiCrxrH-0-F-CwI8bu_UZ_96ro81i"
    # https://drive.google.com/file/d/1_G8TucxHZsZ4YGrpODm4cz7dK4NAbBAa/view?usp=sharing
    snli_10_val_hdf5 = "1_G8TucxHZsZ4YGrpODm4cz7dK4NAbBAa"
    # https://drive.google.com/file/d/1XVEvItusBCCXgshlbgx3bn_fJOUGIOgL/view?usp=sharing
    snli_10_label_hdf5 = "1XVEvItusBCCXgshlbgx3bn_fJOUGIOgL"
    # https://drive.google.com/file/d/116HYEVyzeXDHA4Apyk3pBP4gw_lA4JZZ/view?usp=sharing
    snli_10_word_hdf5 = "116HYEVyzeXDHA4Apyk3pBP4gw_lA4JZZ"
    # https://drive.google.com/file/d/1deYs4h9BeyzvhWplczb8euJvJ2Efcgr1/view?usp=sharing
    nlpcore_model = "1deYs4h9BeyzvhWplczb8euJvJ2Efcgr1"
    # https://drive.google.com/file/d/1UZmw9g7DcpX-aGFeluj_AVoI06yYfz9r/view?usp=sharing
    nlpcore = "1UZmw9g7DcpX-aGFeluj_AVoI06yYfz9r"

    GoogleDriveDownloader.download_file_from_google_drive(file_id=glove_hdf5,
                                        dest_path='data/glove.hdf5')
    GoogleDriveDownloader.download_file_from_google_drive(file_id=local_300_parikh_hdf5,
                                        dest_path='data/local_300_parikh.hdf5')
    GoogleDriveDownloader.download_file_from_google_drive(file_id=local_300_parikh_txt,
                                        dest_path='data/local_300_parikh.txt')
    GoogleDriveDownloader.download_file_from_google_drive(file_id=snli_10_test_hdf5,
                                        dest_path='data/snli_1.0/snli_1.0-test.hdf5')
    GoogleDriveDownloader.download_file_from_google_drive(file_id=snli_10_train_hdf5,
                                        dest_path='data/snli_1.0/snli_1.0-train.hdf5')
    GoogleDriveDownloader.download_file_from_google_drive(file_id=snli_10_val_hdf5,
                                        dest_path='data/snli_1.0/snli_1.0-val.hdf5')
    GoogleDriveDownloader.download_file_from_google_drive(file_id=snli_10_label_hdf5,
                                        dest_path='data/snli_1.0/snli_1.0.label.hdf5')
    GoogleDriveDownloader.download_file_from_google_drive(file_id=snli_10_word_hdf5,
                                        dest_path='data/snli_1.0/snli_1.0.word.hdf5')
    GoogleDriveDownloader.download_file_from_google_drive(file_id=nlpcore_model,
                                        dest_path='data/stanford-corenlp-3.9.0-models.jar')
    GoogleDriveDownloader.download_file_from_google_drive(file_id=nlpcore,
                                        dest_path='data/stanford-corenlp-3.9.0.jar')

    GoogleDriveDownloader.download_file_from_google_drive(
        file_id="1cysjEN9RcsIu9DOwfygeyrFBlQaYpUEy",
        dest_path='data/bidaf/bidaf_5.ema.hdf5')

    # https://drive.google.com/file/d/1puNX0My9W6CDU71ugyvb11fuLTC_GjQZ/view?usp=sharing
    GoogleDriveDownloader.download_file_from_google_drive(
        file_id="1puNX0My9W6CDU71ugyvb11fuLTC_GjQZ",
        dest_path='data/bidaf/bidaf_5.hdf5')

    # https://drive.google.com/file/d/1THLq0fZevBTpv3AsyoVbJEDw3dr-vtCW/view?usp=sharing
    GoogleDriveDownloader.download_file_from_google_drive(
        file_id="1THLq0fZevBTpv3AsyoVbJEDw3dr-vtCW",
        dest_path='data/bidaf/glove.hdf5')

    # https://drive.google.com/file/d/1WvY9JLyA-ytPRaNSv8rcztHj9VJYH5Pw/view?usp=sharing
    GoogleDriveDownloader.download_file_from_google_drive(
        file_id="1WvY9JLyA-ytPRaNSv8rcztHj9VJYH5Pw",
        dest_path='data/bidaf/squad-train.hdf5')

    # https://drive.google.com/file/d/1gAxN4s9zFshC1Wd_aR2NxEvmQqtB7-6w/view?usp=sharing
    GoogleDriveDownloader.download_file_from_google_drive(
        file_id="1gAxN4s9zFshC1Wd_aR2NxEvmQqtB7-6w",
        dest_path='data/bidaf/squad-val.hdf5')

    # https://drive.google.com/file/d/1ysEEtWPVFACWS7TNg_aQgEUNahwNG3m0/view?usp=sharing
    GoogleDriveDownloader.download_file_from_google_drive(
        file_id="1ysEEtWPVFACWS7TNg_aQgEUNahwNG3m0",
        dest_path='data/bidaf/squad.word.dict')

    # https://drive.google.com/file/d/1j3ar-mD3tqkDEat0LOhmLmJ4DTtbPdi6/view?usp=sharing
    GoogleDriveDownloader.download_file_from_google_drive(
        file_id="1j3ar-mD3tqkDEat0LOhmLmJ4DTtbPdi6",
        dest_path='data/bidaf/bidaf_clip5_20.ema.hdf5')

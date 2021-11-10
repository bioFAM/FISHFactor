if __name__ == '__main__':
    import requests
    import zipfile
    import io

    urls = ['https://zenodo.org/record/2669683/files/seqFISH%2B_NIH3T3_point_locations.zip',
            'https://zenodo.org/record/2669683/files/ROIs_Experiment1_NIH3T3.zip',
            'https://zenodo.org/record/2669683/files/ROIs_Experiment2_NIH3T3.zip',
    ]

    for i, url in enumerate(urls):
        r = requests.get(url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall()
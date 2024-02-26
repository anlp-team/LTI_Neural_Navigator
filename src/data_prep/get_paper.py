import numpy as np
import json
import os 
import requests
import time

def get_all_papers(names, save_dir):
    ### expect title, abstract, authors, publication venue, year and tldr

    data = np.genfromtxt(names, delimiter=',', dtype=str)
    ids = []
    for x in data:
        ids.append(x[1])
    # print(ids)
    r = requests.post(
        'https://api.semanticscholar.org/graph/v1/author/batch',
        params={'fields': 'name,hIndex,citationCount,paperCount,affiliations,papers.paperId,papers.openAccessPdf,papers.isOpenAccess,papers.title,papers.publicationVenue,papers.year,papers.abstract,papers.authors'},
        json={"ids":ids}
    )
    # result = json.dumps(r.json(), indent=2)
    with open(save_dir, 'w', encoding='utf-8') as f:
        json.dump(r.json(), f,indent=2)
    # print(result)
    # print('size of json object == ',len(result.encode("utf-8")))

    ### TODO: replace abstract with tldr
    ### TODO: missing LP Morency

def download_pdf_file(url: str, save_path: str,pdf_file_name:str) -> bool:
    """Download PDF from given URL to local directory.

    :param url: The url of the PDF file to be downloaded
    :return: True if PDF file was successfully downloaded, otherwise False.
    """

    # Request URL and get response object
    response = requests.get(url)

    # isolate PDF filename from URL
    if response.status_code == 200:
        # Save in current working directory
        filepath = os.path.join(save_path, pdf_file_name+'.pdf')

        with open(filepath, 'wb') as pdf_object:
            pdf_object.write(response.content)
            print(f'{pdf_file_name} was successfully saved!')
            return True
    else:
        print(f'Uh oh! Could not download {pdf_file_name},')
        print(f'HTTP response status code: {response.status_code}')
        return False

def filter(data_dir, save_dir):
    f = open(data_dir)
    data = json.load(f)
    api_call_counter = 0
    for Prof in data[:4]:
        file_name = Prof['name']
        save_path = os.path.join(save_dir, str(file_name))
        if os.path.exists(save_path):
            file_save_path = save_path+'.txt'
        else:
            os.mkdir(save_path)
        # print(Prof)
        Prof_info=  [Prof['name'],
                     str(Prof['hIndex']),
                     str(Prof['citationCount']),
                     str(Prof['paperCount'])]
        paper_list = []
        for paper in Prof['papers']:
            if paper is not None and paper['year'] == 2023 and paper['isOpenAccess']:
                # print(paper['openAccessPdf']['url'])
                paper = {key: ('' if value is None else value) for key, value in paper.items()}

                authors = [i['name'] for i in paper['authors']]
                # print(authors)
                if type(paper['publicationVenue']) == str:
                    paper['publicationVenue'] = {'name':'', 'alternate_names':''}
                venue_names = []
                venue_names.append(paper['publicationVenue']['name'])
                if type(paper['publicationVenue']['alternate_names']) == str:
                    venue_names.append(paper['publicationVenue']['alternate_names'] )
                else:
                    venue_names+=paper['publicationVenue']['alternate_names'] 
                    
                paper_list.append([paper['title'],
                                   str(paper['year']),
                                #    paper['publicationVenue']['alternate_names']+(paper['publicationVenue']['name']),
                                   venue_names,
                                   paper['abstract'],
                                   authors
                                   ])
            ### save pdf
            if paper['openAccessPdf'] != None:
                if api_call_counter==4:
                    time.sleep(1.1)
                download_pdf_file(url=paper['openAccessPdf']['url'],save_path=save_path, pdf_file_name=paper['title'])
                api_call_counter+=1
                
        with open(file_save_path, "w") as text_file:
            text_file.writelines(Prof_info)
            text_file.write('Papers that are published on 2023 and have open access are listed below with their titles, years, publication venues, as well as the author lists and abstracts')
            for i in paper_list:
                print(i)
                text_file.write(str(i))
        text_file.close()


if __name__ == "__main__":
    filter(data_dir='data.json', save_dir='../../data/Prof_papers/')

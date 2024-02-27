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

def download_pdf_file(api_call_counter, url: str, save_path: str,pdf_file_name:str) -> bool:
    """Download PDF from given URL to local directory.

    :param url: The url of the PDF file to be downloaded
    :return: True if PDF file was successfully downloaded, otherwise False.
    """

    # Request URL and get response object
    filepath = os.path.join(save_path, pdf_file_name+'.pdf')
    if os.path.exists(filepath):
        print(f'{pdf_file_name} has been saved previously!')
        return filepath, api_call_counter
    
    time.sleep(0.2)
    if api_call_counter==4:
        time.sleep(0.8)
        api_call_counter = 0    

    response = requests.get(url, allow_redirects=True, headers={"User-Agent":'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36'})    
    api_call_counter+=1
    if response.status_code == 200:
        # Save in current working directory
        with open(filepath, 'wb') as pdf_object:
            pdf_object.write(response.content)
            print(f'{pdf_file_name} was successfully saved!')
            return filepath, api_call_counter
    else:
        print(f'Uh oh! Could not download {pdf_file_name},')
        print(f'HTTP response status code: {response.status_code}')
        return None,api_call_counter

def filter(data_dir, save_dir):
    ### return a list of relative path to each professor's subdir
    f = open(data_dir)
    data = json.load(f)
    api_call_counter = 0
    all_pdf_paths =[]
    paper_2023_counter = 0
    for Prof in data:
        Prof_name = Prof['name']
        # save_path = os.path.join(save_dir, str(Prof_name))
        save_path = save_dir
        if os.path.exists(save_path):
            Prof_metadata_save_path = save_path+Prof['name']+'.txt'
        else:
            os.mkdir(save_path)
            Prof_metadata_save_path = save_path+Prof['name']+'.txt'
        # print(Prof)
        Prof_info=  [Prof['name'],
                     str(Prof['hIndex']),
                     str(Prof['citationCount']),
                     str(Prof['paperCount'])]
        paper_list = []
        for paper in Prof['papers']:
            
            if paper is not None and paper['year'] == 2023 and paper['isOpenAccess']:
                paper_2023_counter+=1
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

                    if 'pdf' not in paper['openAccessPdf']['url']:
                        url = paper['openAccessPdf']['url']+'.pdf'
                    else:
                        url = paper['openAccessPdf']['url']
                    try:
                        pdf_path,api_call_counter = download_pdf_file(api_call_counter,url=url,save_path=save_path, pdf_file_name=Prof_name+'_'+paper['title'])
                        api_call_counter+=1
                        if pdf_path != None:
                            all_pdf_paths.append(pdf_path)
                    except Exception as e:
                        print(e)
                        continue
                
        with open(Prof_metadata_save_path, "w") as text_file:
            text_file.writelines(Prof_info+"\n")

            text_file.write('Papers that are published on 2023 and have open access are listed below with their titles, years, publication venues, as well as the author lists and abstracts')
            for i in paper_list:
                # print(i)
                text_file.writelines(str(i)+"\n")
        text_file.close()
        print(Prof_name+' has published {} papers*****************'.format(paper_2023_counter))
        paper_2023_counter = 0

    
    return all_pdf_paths


if __name__ == "__main__":
    filter(data_dir='data.json', save_dir='../../data/Prof_papers/')

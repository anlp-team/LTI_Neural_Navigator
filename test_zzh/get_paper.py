import requests
import numpy as np
import json
### expect title, abstract, authors, publication venue, year and tldr
data = np.genfromtxt('Profs_semantic.csv', delimiter=',', dtype=str)
ids = []
for x in data:
    ids.append(x[1])
# print(ids)
r = requests.post(
    'https://api.semanticscholar.org/graph/v1/author/batch',
    params={'fields': 'name,hIndex,citationCount,paperCount,affiliations,papers.paperId,papers.openAccessPdf,papers.isOpenAccess,papers.title,papers.publicationVenue,papers.year,papers.abstract,papers.authors'},
    json={"ids":ids}
)
result = json.dumps(r.json(), indent=2)
with open('data.json', 'w', encoding='utf-8') as f:
    json.dump(r.json(), f,indent=2)
# print(result)
# print('size of json object == ',len(result.encode("utf-8")))

### TODO: replace abstract with tldr
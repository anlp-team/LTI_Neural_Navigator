import requests
from bs4 import BeautifulSoup

# URL of the professor's homepage
url = 'http://professor_homepage.com'

# Send a GET request to the URL
response = requests.get(url)

# Parse the HTML content of the page
soup = BeautifulSoup(response.text, 'html.parser')

# Extract the professor's name, publications, and contact information
# You may need to adjust the selectors based on the actual HTML structure of the page
professor_name = soup.find('tag_for_name', class_='class_for_name').text.strip()
publications = [pub.text.strip() for pub in soup.find_all('tag_for_publications', class_='class_for_publications')]
contact_info = soup.find('tag_for_contact_info', class_='class_for_contact_info').text.strip()

# Save the extracted information to a text file
with open('professor_info.txt', 'w', encoding='utf-8') as file:
    file.write(f"Name: {professor_name}\n")
    file.write("Publications:\n")
    file.write('\n'.join(publications))
    file.write(f"\nContact Information: {contact_info}\n")

print("Information extracted and saved to professor_info.txt")

# MeteoFrance API Client

## Description
This Python project includes a client for interacting with the MeteoFrance API. It allows users to authenticate, retrieve lists of weather stations by department, select a specific station, order and download **hourly** weather data, and view station information. The code is structured around a `Client` class that handles API requests and responses.

## Features
- OAuth2 or token authentication with the MeteoFrance API.
- Retrieve a list of weather stations for a specified department.
- Select and view detailed information about a weather station.
- Order and download weather data for a specified time period.

## Installation
To use this project, clone the repository and install the required dependencies.

```bash
git clone https://github.com/Samy-K/MF_API_CLIENT.git
cd MF_API_CLIENT
pip install -r requirements.txt
```

## Usage
Update the **API_config.txt** file with your MeteoFrance API credentials.
Run the script:
```bash
python main.py
```
Follow the on-screen prompts to select a department, weather station, and data period.

## Dependencies
- **Python 3** (tested with 3.11)
- **requests** - For making HTTP requests to the API.
- **pandas** - For data manipulation and analysis.

## Configuration
Before running the script, ensure you have a valid **API_config.txt** file in your project directory with the following format:
```css
[Parameters]
APPLICATION_ID = Your_Application_ID
DATA_SERVER = MF_API_SERVER
TOKEN = Your_API_Key
```
Note: you must either fill in the application ID (using the OAuth2 method, valid for 1 hour) or the token (validity to be specified when creating).
All information is available on the [Météo France API website](https://portail-api.meteofrance.fr/web/fr).

Information on the data fields of the raw files can be found in this [CSV](https://donneespubliques.meteofrance.fr/client/document/api_clim_table_parametres_horaires_20240103_352.csv)

## Contributing
Contributions to this project are welcome. Please fork the repository and submit a pull request with your proposed changes.

## License
This project is licensed under the Apache-2.0 license. See the LICENSE file for more details.

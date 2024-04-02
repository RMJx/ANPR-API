import requests
import os

def query_dvla_api(registration_plate):
    # Get API key from environment variable
    api_key = os.environ.get('DVLA_API_KEY')
    if not api_key:
        raise ValueError("DVLA_API_KEY environment variable is not set")

    # Construct URL for DVLA API
    url = "https://driver-vehicle-licensing.api.gov.uk/vehicle-enquiry/v1/vehicles"
    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json"
    }
    payload = {
        "registrationNumber": registration_plate
    }

    # Send POST request to DVLA API
    response = requests.post(url, headers=headers, json=payload)

    # Check if request was successful
    if response.status_code == 200:
        # Parse JSON response
        data = response.json()
        return data
    else:
        # Print error message if request failed
        return f"Error: Failed to query DVLA API. Status code: {response.status_code}"


def parseInfo():
    data = query_dvla_api("X99RMJ")

    print("Make: " + data.get('make', 'N/A'))
    print("Model: " + data.get('model', 'N/A'))
    print("artEndDate: " + data.get('artEndDate', 'N/A'))
    print("Colour: " + data.get('colour', 'N/A'))
    print("Engine Capacity: " + str(data.get('engineCapacity', 'N/A')))
    print("First Registered: " + data.get('monthOfFirstRegistration', 'N/A'))
    print("Tax Due: " + data.get('taxDueDate', 'N/A'))
    print("Tax Status: " + data.get('taxStatus', 'N/A'))
    print("Year: " + str(data.get('yearOfManufacture', 'N/A')))
    print("Euro Status: " + data.get('euroStatus', 'N/A'))
    print("CO2 Emissions: " + str(data.get('realDrivingEmissions', 'N/A')))
    print("V5C Date: " + data.get('dateOfLastV5CIssued', 'N/A'))
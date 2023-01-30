# SADS_OFFLINE
## Description
Offline Shop-floor Anomaly Detection System ( OfF-SADS ) 

## Running the solution:
- [x] Runing using docker-compose in detach mode
    - docker-compose up --build -d 
- [x] Access the solution on the port 8080 ( localhost:8080)
## Eroor handling 
- [x] If you ancounter the following error for older docker version (1.8.0 and older)
    -   ERROR: The compose file './docker-compose.yml' is invalid because: Unsuported config option for service: 'app'
        Creating sads_batch ... error

    -   Set the version to correct one at the top of the docker-compose.yml file: version: '2.0'

# SADS_OFFLINE
## Description
Offline Shop-floor Anomaly Detection System ( OfF-SADS ) 

## Running the solution:
- [x] Runing using docker-compose in detach mode
    - docker-compose up --build -d 
- [x] Access the solution on the port 8080 ( localhost:8080)
## Eroor handling 
- [x] If you ancounter the following error 
    -   Successfully tagged sads_offline_app:latest
        Creating sads_batch ... error

        ERROR: for sads_batch  Cannot create container for service app: Conflict. The container name "/sads_batch" is already in use by container "77bbd38e393cbd4ffd22605be53b9252c23a3e009afb9c4caadcc9608b341bbf". You have to remove (or rename) that container to be able to reuse that name.

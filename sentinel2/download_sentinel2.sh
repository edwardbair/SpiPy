#!/bin/zsh
#run STAC_query.sh first to create files.txt
for line in "${(@f)"$(<files.txt)"}"
{
#get access token
curl --location --request POST 'https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token' \
--header 'Content-Type: application/x-www-form-urlencoded' \
--data-urlencode 'grant_type=password' \
--data-urlencode 'username=[user]' \
--data-urlencode 'password=[password]' \
--data-urlencode 'client_id=cdse-public' -o token.txt
#read access token
ACCESS_TOKEN=$(jq '.access_token' token.txt | sed 's/\"//g')
#create output filename from item #
fname=$(echo $line | awk -F'[()]' '{print $2}')
#create curl string
curl_str="curl -H \"Authorization: Bearer ${ACCESS_TOKEN}\" ${line} --location-trusted --output ${fname}.zip"
#evaluate curl string
eval $curl_str
}
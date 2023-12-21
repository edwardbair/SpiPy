#!/bin/sh

#queries and creates list of Sentinel-2 files
url="https://catalogue.dataspace.copernicus.eu/stac/collections"
collection="SENTINEL-2"
datetimestart="2022-10-01T00:00:00.000Z"
datetimeend="2023-07-01T00:00:00.000Z"
bbox="-119,37,-118,38"
producttype="S2MSI2A"
tile="11SLB"

curlstr="${url}/${collection}/items?datetime=${datetimestart}/${datetimeend}&bbox=${bbox}&limit=1000&sortby=-end_datetime"
jqstr=\'".features[] | select(.properties.productType == \"${producttype}\") | select(.properties.tileId==\"${tile}\") | .assets.PRODUCT.href"\'

str="curl \"$curlstr\" | jq $jqstr | sed \"s/\\\"/\'/g\" > files.txt"

eval $str
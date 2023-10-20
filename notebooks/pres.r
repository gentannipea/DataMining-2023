# load a rda
pres_by_cd <= load("DataMining-2023/dataset/congress/pres_by_cd.rda")
View(pres_by_cd)

pres_by_county = 'dsa'
pres_by_county <= load("DataMining-2023/dataset/congress/pres_by_county.rda")
View(pres_by_county)

# write pres_by_county to csv
write.csv(pres_by_county, "DataMining-2023/dataset/congress/pres_by_county.csv")

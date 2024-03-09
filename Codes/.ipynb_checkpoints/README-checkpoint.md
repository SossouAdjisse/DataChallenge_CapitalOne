

### Cleaning and Restrictions: 
**Flights dataset:**
1. Converted FL_DATE to YYYY-MM-DD format 
2. Converted  AIR_TIME and DISTANCE from objects format to float64 format
3. Dropped the 4545 duplicates with respect to all the columns 
4. Replaced missing values in columns with the median (because of outliers) in columns 'DEP_DELAY', 'ARR_DELAY', 'AIR_TIME', 'DISTANCE', and ‘OCCUPANCY_RATE'
5. Dropped out 12111  rows where TAIL_NUM is missing
6. Dropped out 39368 rows where the ‘flight is cancelled (CANCELLED' == 1)
    - Final data: flights_clean

**Tickets dataset:**
1. Converted ITIN_FARE from string to float64 format
2. Dropped out  71898 duplicates with respect to all the columns 
3. Imputed missing values in columns 'PASSENGERS', and ‘ITIN_FARE' with the median because of outliers
4. Kept only the 661036 rows of round tickets, dropping out 434351 rows where the column ROUNDTRIP == 0
    - Final data: tickets_clean

**Airports Codes dataset:**
1. Dropped the 101 duplicates with respect to all the columns
2. Kept only the US airport IATA CODES, meaning keeping rows where ISO_COUNTRY is equal to 'US’.  This ensures we are only dealing with US domestic airports
3. Filtered only  rows where TYPE is 'medium_airport' or 'large_airport'
4. Dropped out 37 invalid  rows where 'IATA_CODE' is missing
    - Final data: airport_codes_clean
    
    
 
 ### Steps for Aggreagte and Merging the flights, tickets, and airports codes datasets
**Filter out US domestic flights on both ORIGIN and DESTINATION:**
1. Merge the flights_clean and airport_codes_clean with ORIGIN and IATA_CODE as keys on left and right respectively using inner join. This ensures the origin flights are US domesic market only because I made sure the IATA CODES in airport_codes_clean are all with the US.
    - Resulting data: **flights_airportcodes_merge1**
2. Merge the **flights_airportcodes_merge1** above and airport_codes_clean with DESTINATION and IATA_CODE as keys on left and right respectively using inner join. This ensures the destination flights are US domesic market only
      - Resulting data: **flights_airportcodes_merge2**
3. Create the routes identifiers called ROUTE_ID
**Filter out US domestic tickets on DESTINATION:**
1. Merge the tickets_clean and airport_codes_clean with DESTINATION and IATA_CODE as keys on left and right respectively using inner join. This ensures the destination tickets are US domesic market only.
      - Resulting data: **tickets_airportcodes_merge3**
2. **_Commnent_:** I don't need to filter the tickets at the roigin because the column ORIGIN_COUNTRY in the tickets dataset ensured me that the origin of all the tickets is the US.

**Aggregate the tickects_clean:**
1. I will create the route identifiers ROUTE_ID in **tickets_airportcodes_merge3**
2. I will aggregate the **tickets_airportcodes_merge3** on the total ROUNDTRIP, PASSENGERS, and ITIN_FARE for every ROUTE_ID
      - Resulting data: **tickets_airportcodes_merge3_aggreg**

Finally, I will merge **flights_airportcodes_merge2** and **tickets_airportcodes_merge3_aggreg** using ROUTE_ID as the key and an inner join.



### Imperfections in the datasets and limitations of the above merging approach:
1. The ideal would be to connect every ticket to its flight using the tickets_clean and flights_clean data before aggregating total ROUNDTRIP, PASSENGERS, and ITIN_FARE within ROUTE_ID in the tickets_clean
2. But with the provided information, I failed to connect evey ticket to its flight
3. I tried hard to extract and combine information from other columns to do that, but I failed at this points.
4. For example, I attempted to break ITIN_ID column down to extract some information but hit a wall. 
4. I searched  many open-source resources but did not succeed so far to find helpful information.
5. So, the main flaw in my merging method is the following:
    - aggreagting total ROUNDTRIP, PASSENGERS, and ITIN_FARE with ROUTE_ID in the tickets_clean might be compromised by cacelled flights information.
    - However, any ROUTE_ID in tickets_clean where all the flights have been cancelled will be dropped after merging with flights_clean. This is good for me as it will improve the accuracy of my analysis.  
    - **For now, I assume that every tickets in the tickets_clean data is on a flight that was not called.** 
6. If I have more time, I would further my search on how to link the tickets to the flights. 


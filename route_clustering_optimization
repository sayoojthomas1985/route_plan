def run_main_program(num_drivers=None, delivery_date=None):
   #!/usr/bin/env python
    # coding: utf-8
    import requests
    from sqlalchemy import create_engine
    from datetime import datetime
    import pandas as pd
    import math
    from sklearn.cluster import KMeans
    from haversine import haversine
    import warnings
    import time
    import os
    from dotenv import load_dotenv
    from collections import Counter
    import math
    warnings.simplefilter("ignore", FutureWarning)  

    def normalize_date_to_string(date_obj):
       '''Convert various date formats to YYYY-MM-DD string'''
       if date_obj is None:
           return None
       if isinstance(date_obj, str):
           return date_obj[:10]
       if hasattr(date_obj, 'strftime'):
           return date_obj.strftime('%Y-%m-%d')
       return str(date_obj)[:10]
    
    # = Load environment variables =
    load_dotenv()
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = int(os.getenv("DB_PORT"))
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_NAME = os.getenv("DB_NAME")
    
       
    # = SQLAlchemy Engine =
    engine = create_engine(
        f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    
    sessions = ["Breakfast", "Lunch", "Dinner"]
    route_plan_files = {}
    map_links_files = {}
    
    # Configuration constants
    MAX_POINTS = 20
    MAX_TIME = 120
    MIN_TIME = 90
    MAX_TIME_ABS = 130
    TARGET_AVG_TIME = 104.5
    AVERAGE_SPEED = 22.5  # km/h
    DEPOT = (10.0352754, 76.4100184)
    MAX_ATTEMPTS = 600
    MAX_WAYPOINTS = 23
    SERVICE_TIME_MIN = 5

    if num_drivers is None:
        num_drivers = get_num_drivers()
    
    # Helper functions
    def compute_distance(c1, c2):
        return haversine(c1, c2)
    
    def route_distance(coords):
        dist = 0
        last = DEPOT
        for pt in coords:
            dist += compute_distance(last, pt)
            last = pt
        dist += compute_distance(last, DEPOT)
        return dist
    
    def calculate_time_km(points):
        if not points: 
            return 0, 0
        coords = [x[0] for x in points]
        dist = route_distance(coords)
        time = dist / AVERAGE_SPEED * 60
        return round(dist, 1), round(time, 1)
    
    def cluster_summary(df, phase):
        summary = []
        for c_id, group in df.groupby('Cluster'):
            pts = list(zip(group['Coords'], [1]*len(group)))
            dist, time = calculate_time_km(pts)
            overloaded = time > MAX_TIME_ABS or len(group) > MAX_POINTS
            underused = time < MIN_TIME
            packages_sum = group['Packages'].fillna(0).sum() if 'Packages' in group else len(group)
            summary.append({
                'Phase': phase,
                'Cluster_ID': c_id,
                'Points': len(group),
                'Packages': packages_sum,
                'KMs': dist,
                'Time_Minutes': time,
                'Overloaded': overloaded,
                'Underused': underused
                    })
        return pd.DataFrame(summary)
    
    def initial_cluster_fixed_k(df, k_drivers):
        # Drop rows with missing latitude or longitude
        valid_df = df.dropna(subset=['Final_Latitude', 'Final_Longitude']).copy()

        # Cluster only valid customers
        if not valid_df.empty:
            km = KMeans(n_clusters=k_drivers, random_state=1).fit(
             valid_df[['Final_Latitude', 'Final_Longitude']]
            )
            valid_df['Cluster'] = km.labels_
        return valid_df

    def transfer_points_fixed(df, from_cluster, to_cluster, max_moves=12):
        high_df = df[df['Cluster'] == from_cluster]
        low_df = df[df['Cluster'] == to_cluster]
        if high_df.empty or low_df.empty:
            return df, False    
        distances = [
            (idx, compute_distance(row['Coords'], low_df['Coords'].iloc[0]))
            for idx, row in high_df.iterrows()
        ]
        distances.sort(key=lambda x: x[1])    
        moved = False
        for move_count in range(1, min(max_moves, len(distances)) + 1):
            to_move = [idx for idx, _ in distances[:move_count]]
            temp_df = df.copy()
            temp_df.loc[to_move, 'Cluster'] = to_cluster
            new_summary = cluster_summary(temp_df, "Test_Move")    
            if new_summary[new_summary['Cluster_ID'] == to_cluster]['Time_Minutes'].iloc[0] <= MAX_TIME_ABS:
                df = temp_df
                moved = True
                break    
        return df, moved
    
    def rebalance_fixed(df, num_drivers):
        for attempt in range(MAX_ATTEMPTS):
            clusters = cluster_summary(df, 'After')
            avg_time = clusters['Time_Minutes'].mean()
            if all((~clusters['Overloaded']) & (~clusters['Underused'])):
                return df    
            overloaded_clusters = clusters[clusters['Overloaded']]['Cluster_ID'].tolist()
            underused_clusters = clusters[clusters['Underused']]['Cluster_ID'].tolist()    
            for oid in overloaded_clusters:
                if not underused_clusters: 
                    break
                for uid in underused_clusters:
                    df, moved = transfer_points_fixed(df, oid, uid)
                    if moved:
                        break
        return df
    
    def split_until_balanced_fixed(df, num_drivers):
        while True:
            summary = cluster_summary(df, "Split_Check")
            overloaded = summary[
                ((summary['Time_Minutes'] > MAX_TIME) | (summary['Points'] > MAX_POINTS))
                & (summary['Time_Minutes'] >= MIN_TIME)
            ]
            if overloaded.empty:
                break    
            underused = summary[summary['Time_Minutes'] < MIN_TIME]['Cluster_ID'].tolist()
            if not underused:
                break    
            for oid in overloaded['Cluster_ID']:
                for uid in underused:
                    df, moved = transfer_points_fixed(df, oid, uid)
                    if moved:
                        break
        return df
    
    def strict_balance(df, max_loops=60):
        for loop in range(max_loops):
            summary = cluster_summary(df, f"Strict_{loop+1}")
            max_time = summary['Time_Minutes'].max()
            min_time = summary['Time_Minutes'].min()
            avg_time = summary['Time_Minutes'].mean()
    
            changed = False
    
            overloaded = summary[
                ((summary['Time_Minutes'] > MAX_TIME_ABS) | (summary['Points'] > MAX_POINTS))
                & (summary['Time_Minutes'] >= MIN_TIME)
            ]
            underused = summary[summary['Time_Minutes'] < MIN_TIME]['Cluster_ID'].tolist()
    
            for cid in overloaded['Cluster_ID']:
                # Instead of splitting, move points to underused clusters if possible
                for uid in underused:
                    df, moved = transfer_points_fixed(df, cid, uid)
                    if moved:
                        changed = True
                        break
    
            if not changed:
                return df    
        return df
    
    def cascade_rebalance(df, max_loops=40, max_dev_allowed=40):
        for loop in range(max_loops):
            summary = cluster_summary(df, f"Cascade_{loop+1}")
            summary = summary.sort_values(by="Time_Minutes", ascending=False)
            max_time = summary['Time_Minutes'].iloc[0]
            min_time = summary['Time_Minutes'].iloc[-1]
            max_dev = max_time - min_time
    
            if max_dev <= max_dev_allowed:
                break
    
            moved_any = False
            high_cluster = summary['Cluster_ID'].iloc[0]
            underused_clusters = summary['Cluster_ID'].iloc[1:].tolist()
            for low_cluster in underused_clusters:
                df, moved = transfer_points_fixed(df, high_cluster, low_cluster, max_moves=3)
                if moved:
                    moved_any = True
                    break
    
            if not moved_any:
                break
    
        return df
    
    def fix_final_imbalance_best(df_original):
        best_df = df_original.copy()
        best_score = float('inf')
    
        for max_move in range(1, 24):
            df = df_original.copy()
            clusters = cluster_summary(df, "Final")
            overloaded = clusters[clusters['Overloaded']]['Cluster_ID'].tolist()
            underused = clusters[clusters['Underused']]['Cluster_ID'].tolist()
    
            for oid in overloaded:
                for uid in underused:
                    df, moved = transfer_points_fixed(df, oid, uid, max_moves=max_move)
                    if moved:
                        break
    
            summary = cluster_summary(df.copy(), f"MAX_MOVE={max_move}")
            time_diffs = abs(summary["Time_Minutes"] - 150)
            score = time_diffs.sum()
    
            if score < best_score:
                best_score = score
                best_df = df.copy()
    
        return best_df
    
    # Add route optimization methods here (nearest_neighbor_route, two_opt, selective_three_opt, selective_four_opt, hybrid_optimize)
          
    def nearest_neighbor_route(coords):
        """Basic nearest neighbor route for initial path."""
        if not coords:
            return []
        
        unvisited = list(coords)
        route = [unvisited.pop(0)]
        while unvisited:
            next_point = min(unvisited, key=lambda x: compute_distance(route[-1], x))
            route.append(next_point)
            unvisited.remove(next_point)
        return route
          
    def two_opt(route):
        best = route
        improved = True
        while improved:
            improved = False
            for i in range(1, len(best) - 2):
                for j in range(i + 1, len(best)):
                    if j - i == 1: 
                        continue
                    new_route = best[:i] + best[i:j][::-1] + best[j:]
                    if route_distance(new_route) < route_distance(best):
                        best = new_route
                        improved = True
            route = best
        return best  
    
    def exhaustive_three_opt(route):
        best = route
        best_distance = route_distance(route)
        n = len(route)
        improved = True
        while improved:
            improved = False
            for i in range(1, n-2):
                for j in range(i+1, n-1):
                    for k in range(j+1, n):
                        seg1, seg2, seg3, seg4 = route[:i], route[i:j], route[j:k], route[k:]
                        candidates = [
                            seg1 + seg2[::-1] + seg3 + seg4,
                            seg1 + seg2 + seg3[::-1] + seg4,
                            seg1 + seg3 + seg2 + seg4,
                            seg1 + seg3[::-1] + seg2[::-1] + seg4,
                            seg1 + seg3[::-1] + seg2 + seg4,
                            seg1 + seg2[::-1] + seg3[::-1] + seg4,
                                    ]
                        for cand in candidates:
                            d = route_distance(cand)
                            if d < best_distance:
                                best = cand
                                best_distance = d
                                improved = True
            route = best
        return best
    
    def exhaustive_four_opt(route):
        best = route
        best_distance = route_distance(route)
        n = len(route)
        improved = True
        while improved:
            improved = False
            for i in range(1, n-3):
                for j in range(i+1, n-2):
                    for k in range(j+1, n-1):
                        for l in range(k+1, n):
                            seg1, seg2, seg3, seg4, seg5 = route[:i], route[i:j], route[j:k], route[k:l], route[l:]
                            # There are dozens of possible 4-opt moves. Here's a couple as example:
                            candidates = [
                                seg1 + seg2[::-1] + seg3 + seg4[::-1] + seg5,
                                seg1 + seg3 + seg2 + seg4 + seg5,
                                seg1 + seg3[::-1] + seg2[::-1] + seg4 + seg5,
                                seg1 + seg2[::-1] + seg3[::-1] + seg4[::-1] + seg5
                                # You may expand this list as needed for true exhaustiveness.
                            ]
                            for cand in candidates:
                                d = route_distance(cand)
                                if d < best_distance:
                                    best = cand
                                    best_distance = d
                                    improved = True
            route = best
        return best
    
    def selective_three_opt(route):
        """Selective 3-opt swaps for speed."""
        best = route
        best_distance = route_distance(route)
        n = len(route)
        improved = True
        while improved:
            improved = False
            for i in range(0, n-4, 2):
                for j in range(i+2, n-2, 2):
                    for k in range(j+2, n, 2):
                        A, B, C = route[i:j], route[j:k], route[k:]
                        new_routes = [
                            route[:i] + A[::-1] + B + C,
                            route[:i] + A + B[::-1] + C,
                            route[:i] + B + A + C
                            ]
                        for nr in new_routes:
                            d = route_distance(nr)
                            if d < best_distance:
                                best = nr
                                best_distance = d
                                improved = True
            route = best
        return best   
    
    def selective_four_opt(route):
        """Selective 4-opt swaps."""
        best = route
        best_distance = route_distance(route)
        n = len(route)
        for i in range(0, n-5, 3):
            for j in range(i+2, n-3, 3):
                for k in range(j+2, n-1, 3):
                    for l in range(k+2, n, 3):
                        new_route = route[:i] + route[i:j][::-1] + route[j:k][::-1] + route[k:l][::-1] + route[l:]
                        d = route_distance(new_route)
                        if d < best_distance:
                            best = new_route
                            best_distance = d
        return best  
    
    def hybrid_optimize(route):
        """Hybrid optimization: 2-opt + selective 3-opt + selective 4-opt."""
        best_route = two_opt(route)
        best_route = exhaustive_three_opt(best_route)
        best_route = exhaustive_four_opt(best_route)
        return best_route
          
    # Also add create_google_maps_links as per your original code
    def create_google_maps_links(coords_list):
        urls = []  
        start = 0
        while start < len(coords_list):
            end = min(start + MAX_WAYPOINTS, len(coords_list))
            chunk = coords_list[start:end]
            waypoints = "/".join([f"{lat},{lng}" for lat, lng in chunk])
            if start == 0:
                url = f"https://www.google.com/maps/dir/{DEPOT[0]},{DEPOT[1]}/{waypoints}"
            else:
                url = f"https://www.google.com/maps/dir/{coords_list[start-1][0]},{coords_list[start-1][1]}/{waypoints}"
            if end == len(coords_list):
                url += f"/{DEPOT[0]},{DEPOT[1]}"
            urls.append(url)
            start += MAX_WAYPOINTS
        return urls    
    
    # Placeholder implementations for those functions to be filled exactly as in your original script
    def optimize_route_with_duplicates(coords):
        """Optimize route while handling duplicate coordinates."""
        if not coords:
            return []

        coord_counts = Counter(coords)

        # Optimize only unique points
        unique_coords = list(coord_counts.keys())
        route = nearest_neighbor_route(unique_coords)
        route = hybrid_optimize(route)
    
        # Reinsert duplicates in order
        expanded_route = []
        for c in route:
            expanded_route.extend([c] * coord_counts[c])
        return expanded_route

    def deduplicate_maplinks(df):
        # Only run if Route and Map Link columns exist
        if "Route" not in df.columns or "Map Link" not in df.columns:
            return df      
        cleaned_dfs = []
        for route, g in df.groupby("Route"):
            # Collect unique links for the route
            unique_links = list(dict.fromkeys(g["Map Link"].dropna().tolist()))  # preserves order    
            # Rule: If only one unique link → keep one, if two → keep two, etc.
            if len(unique_links) <= 1:
                g["Map Link"] = unique_links[0] if unique_links else None
            else:
                # Assign links sequentially per stop group of max 23 stops
                assigned_links = []
                for i, row in enumerate(g.itertuples(index=False), start=1):
                    link_index = (i - 1) // 23
                    if link_index < len(unique_links):
                        assigned_links.append(unique_links[link_index])
                    else:
                        assigned_links.append(unique_links[-1])  # fallback last
                g = g.copy()
                g["Map Link"] = assigned_links    
            cleaned_dfs.append(g)    
        return pd.concat(cleaned_dfs, ignore_index=True)   

    def sanitize_for_json(obj):
        if isinstance(obj, float) and math.isnan(obj):
            return None
        elif isinstance(obj, dict):
            return {k: sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize_for_json(item) for item in obj]
        else:
            return obj
               
    all_sessions_json = {}
    for session in sessions:
        print(f"\n🔎 Processing session: {session}")
       
        # If delivery_date is provided, use it; else use tomorrow
        if delivery_date:
            delivery_date_str = normalize_date_to_string(delivery_date)
        else:
            delivery_date_str = None  # will use CURDATE() + INTERVAL 1 DAY in SQL
       
        # Parameterized SQL query
        QUERY = """
        SELECT 
            di.order_id,
            di.quantity AS Packages,
            di.delivery_date AS Date,
            di.delivery_time_slot AS Session,
            mi.name AS menu_item_name,
            c.first_name,
            c.last_name,
            c.whatsapp_number,
            a.id AS address_id,
            a.housename,
            a.street,
            a.geo_location
        FROM delivery_items di
        JOIN addresses a ON di.address_id = a.id
        JOIN contacts c ON di.user_id = c.user_id
        JOIN menu_items mi ON di.menu_item_id = mi.id
        WHERE di.delivery_time_slot = %s
        """
       
        params = [session]
   
        QUERY += " AND di.delivery_date = %s"
        params.append(delivery_date_str)
   
        QUERY += " ORDER BY di.delivery_time_slot, di.delivery_date, di.id;"
   
        df = pd.read_sql(QUERY, engine, params=tuple(params))
        print(f"✅ Loaded {len(df)} deliveries for {session}")
          
        if df.empty:
            continue
    
        # --- Parse geo_location into latitude/longitude ---
        # --- Parse geo_location into latitude/longitude ---
        df[['Final_Latitude', 'Final_Longitude']] = df['geo_location'].str.split(',', expand=True).astype(float)
        # Drop rows where latitude or longitude is missing to exclude incomplete entries
        df = df.dropna(subset=['Final_Latitude', 'Final_Longitude']).reset_index(drop=True)

         
        # Drop rows where lat or lon is missing
        df = df.dropna(subset=['Final_Latitude', 'Final_Longitude']).reset_index(drop=True)
         
        # --- Create Coords column (tuple for clustering/routing later) ---
        df['Coords'] = df.apply(lambda x: (x['Final_Latitude'], x['Final_Longitude']), axis=1)

    
        # --- Rename columns for clarity ---
        df.rename(columns={
            'quantity': 'Packages',
            'first_name': 'First_Name',
            'last_name': 'Last_Name',
            'whatsapp_number': 'Contact',
            'housename': 'HouseName',
            'street': 'Street',
            'menu_item_name': 'MenuItem'
        }, inplace=True)
        
        # --- Get active delivery executives ---
        query = """
        SELECT 
            CONCAT(c.first_name, ' ', c.last_name, ' (', c.whatsapp_number, ')') AS exec_name,
            u.id AS user_id
        FROM user_roles ur
        JOIN users u 
            ON ur.user_id = u.id
        JOIN contacts c 
            ON u.id = c.user_id
        WHERE ur.name = 'DELIVERY_EXECUTIVE'
          AND u.status = 'ACTIVE'
        ORDER BY u.id;
        """
        exec_df = pd.read_sql(query, engine)
        
        # List and mapping of executives
        EXECUTIVE_NAMES = exec_df['exec_name'].tolist()
        EXECUTIVE_MAP = dict(zip(exec_df['exec_name'], exec_df['user_id']))

        num_customers = len(df)
        if num_customers <= 4:
            print(f"⚡ Only {num_customers} customers → skipping clustering, assigning to one executive.")
            df["Cluster"] = 0
            # Assign to first available executive
            df["Executive"] = EXECUTIVE_NAMES[0] if len(EXECUTIVE_NAMES) > 0 else "Executive_1"
            df_missing = pd.DataFrame()  # no missing
            df_final = df.copy()
        else:
            # --- Ensure clusters < customers ---
            if num_drivers >= num_customers:
                num_drivers = num_customers - 1
                    
            # --- Clustering and balancing ---
            df_clustered = initial_cluster_fixed_k(df.copy(), num_drivers)
            df_rebalanced = rebalance_fixed(df_clustered.copy(), num_drivers)
            df_rebalanced = split_until_balanced_fixed(df_rebalanced.copy(), num_drivers)
            df_strict_balanced = strict_balance(df_rebalanced.copy())
            df_cascade_balanced = cascade_rebalance(df_strict_balanced.copy())
            df_final = fix_final_imbalance_best(df_cascade_balanced.copy())
        
        # --- Ensure coordinates column exists ---
        if "Coords" not in df_final.columns:
            df_final["Coords"] = list(zip(df_final["Final_Latitude"], df_final["Final_Longitude"]))
        df_final["Coords"] = df_final["Coords"].apply(lambda x: eval(str(x)) if isinstance(x, str) else x)
        
        # --- Prepare Excel rows ---
        excel_rows = []
        
        for driver_num, (cluster_id, group) in enumerate(df_final.groupby("Cluster"), 1):
            executive_name = EXECUTIVE_NAMES[(driver_num - 1) % len(EXECUTIVE_NAMES)]
            group = group.copy()
            
            coords_list = list(group["Coords"])
            optimized_coords = optimize_route_with_duplicates(coords_list)
            
            # Map coordinates to rows
            optimized_rows = []
            temp_group = group.copy()
            for c in optimized_coords:
                match_idx = temp_group.index[temp_group["Coords"] == c][0]
                optimized_rows.append(temp_group.loc[match_idx])
                temp_group = temp_group.drop(match_idx)
        
            # Compute distances
            cumulative_km = 0
            cumulative_time_min = 0  
            prev_coord = DEPOT
            
            # Google Maps links for this cluster
            map_links = create_google_maps_links(optimized_coords)
            map_link_str = ", ".join(map_links)  # multiple links if needed
        
            for stop_num, row in enumerate(optimized_rows, 1):
                curr_coord = row["Coords"]
                leg_km = compute_distance(prev_coord, curr_coord)
                cumulative_km += leg_km
                if leg_km == 0:
                    leg_time_min = SERVICE_TIME_MIN
                else:
                    leg_time_min = round((leg_km / AVERAGE_SPEED) * 60, 1) + SERVICE_TIME_MIN
                    cumulative_time_min += leg_time_min
        
                excel_rows.append({
                        "Date": row["Date"],
                        "Cluster": cluster_id,
                        "Executive": executive_name,
                        "Stop_No": stop_num,
                        "Delivery_Name": f"{row['First_Name']} {row['Last_Name']}",
                        "Location": f"{row['HouseName']}, {row['Street']}",
                        "Latitude": row['Final_Latitude'],
                        "Longitude": row['Final_Longitude'],
                        "Distance_From_Prev_Stop_km": round(leg_km, 2),
                        "Cumulative_Distance_km": round(cumulative_km, 2),
                        "Leg_Time_min": leg_time_min,                      # ✅ fixed
                        "Cumulative_Time_min": round(cumulative_time_min, 1),  # ✅ fixed
                        "Packages": row["Packages"],                       # ✅ also fix, your DataFrame has `Packages`, not `packages`
                        "Map_Link": create_google_maps_links([prev_coord, curr_coord])[0],
                        })
                prev_coord = curr_coord
        
            # Return to Hub
            leg_km = compute_distance(prev_coord, DEPOT)
            cumulative_km += leg_km
            leg_time_min = round((leg_km / AVERAGE_SPEED) * 60, 1)
            cumulative_time_min += leg_time_min
            excel_rows.append({
                        "Date": row["Date"],   # or use `df['Date'].iloc[0]` if you want session-level date
                        "Cluster": cluster_id,
                        "Executive": executive_name,
                        "Stop_No": len(optimized_rows) + 1,
                        "Delivery_Name": "Return to Hub",
                        "Location": "",
                        "Latitude": DEPOT[0],
                        "Longitude": DEPOT[1],
                        "Distance_From_Prev_Stop_km": round(leg_km, 2),
                        "Cumulative_Distance_km": round(cumulative_km, 2),
                        "Leg_Time_min": leg_time_min,                     # ✅ fixed
                        "Cumulative_Time_min": round(cumulative_time_min, 1), # ✅ fixed
                        "Packages": "",
                        "Map_Link": create_google_maps_links([prev_coord, DEPOT])[0]
                    })
        for idx, row in enumerate(excel_rows):
            if not isinstance(row, dict):
                raise ValueError(f"excel_rows[{idx}] is not a dict: {row!r}")
            for key, val in row.items():
                if isinstance(val, list):
                    raise ValueError(f"excel_rows[{idx}]['{key}'] is a list: {val!r}")

        
        # --- Save Excel report ---
        excel_df = pd.DataFrame(excel_rows)
        excel_df = deduplicate_maplinks(excel_df)
        
        if "Cluster" in excel_df.columns:
            excel_df = excel_df.drop(columns=["Cluster"])
        
        # Cluster summary totals
        summary_rows = []
        for cluster_id, group in excel_df.groupby("Executive"):   # 👈 group by Executive instead
            total_dist = group["Cumulative_Distance_km"].max()
            summary_rows.append({
                "Date": row["Date"],
                "Executive": group["Executive"].iloc[0],
                "Total_Stops": group["Stop_No"].max(),
                "Total_Distance_km": total_dist,
                "Estimated_Time_min": round((total_dist / AVERAGE_SPEED) * 60, 1)
                })
        summary_df = pd.DataFrame(summary_rows)
        # Drop Cluster column if still present
        if "Cluster" in summary_df.columns:
            summary_df = summary_df.drop(columns=["Cluster"])
        final_df = excel_df        
        # --- Save Excel: ONLY one sheet ---
        excel_file = f"/home/ubuntu/route_plan_{session.lower()}.xlsx"
        with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
            final_df.to_excel(writer, sheet_name="Route Plan", index=False)
        
            
        # Inside the loop (replacing the old return line):
        print(f"✅ Excel report saved: {excel_file}")
        final_df['Date'] = final_df['Date'].apply(lambda x: normalize_date_to_string(x))
        final_df = final_df.where(pd.notnull(final_df), None)        
        records = final_df.to_dict(orient="records")
        all_sessions_json[session.lower()] = sanitize_for_json(records)

        
        
        # After the for loop ends (outside the loop):
    return all_sessions_json

        
if __name__ == "__main__":
    run_main_program(num_drivers=2)
    """import json
    all_sessions_json = run_main_program(num_drivers=2)
    for session, records in all_sessions_json.items():
        for rec in records:
            if 'Date' in rec and hasattr(rec['Date'], 'isoformat'):
                rec['Date'] = rec['Date'].isoformat()  # YYYY-MM-DD
    print(json.dumps(all_sessions_json, indent=4))"""

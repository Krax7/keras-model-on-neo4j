
import keras
import numpy as np
import more_itertools
import json
from neo4j.v1 import GraphDatabase, Driver

class GraphSequence(keras.utils.Sequence):

	def __init__(self, args, batch_size=32, test=False):
		self.batch_size = batch_size

		# Setting query to get all the users that reviewed an airbnb
		self.query = """
			MATCH p=
				(u:User)
					-[:WROTE]->
				(r:Review)
					-[:REVIEWS]->
				(l:Listing) WHERE EXISTS(l.review_scores_value)
			RETURN 
				u.user_id AS user,
				l.availability_365 AS av365,
				l.availability_90 AS av90,
				l.availability_60 AS av60,
				l.availability_30 AS av30,
				l.cleaning_fee AS cleaning_fee,
				l.security_deposit AS security_deposit,
				l.monthly_price AS monthly_price,
				l.weekly_price AS weekly_price,
				l.square_feet AS square_feet,
				l.beds AS num_beds,
				l.bedrooms AS num_bedrooms,
				l.bathrooms AS num_bathrooms,
				l.accommodates AS accommodates,
				l.price AS price,
				CASE WHEN l.review_scores_value > 9 THEN 1 ELSE 0 END AS y;
		"""

		self.query_params = {
			"dataset_name": "airbnb",
			"test": test
		}

		with open('./settings.json') as f:
			self.settings = json.load(f)[args.database]

		driver = GraphDatabase.driver(
			self.settings["neo4j_url"], 
			auth=(self.settings["neo4j_user"], self.settings["neo4j_password"]))

		with driver.session() as session:
			data = session.run(self.query, **self.query_params).data()
			data = [ (np.array([
				int(i["user"]),
				i["av365"] if i["av365"] is not None else 0,
				i["av90"] if i["av90"] is not None else 0,
				i["av60"] if i["av60"] is not None else 0,
				i["av30"] if i["av30"] is not None else 0,
				i["cleaning_fee"] if i["cleaning_fee"] is not None else 0,
				i["security_deposit"] if i["security_deposit"] is not None else 0,
				i["monthly_price"] if i["monthly_price"] is not None else 0,
				i["weekly_price"] if i["weekly_price"] is not None else 0,
				i["square_feet"] if i["square_feet"] is not None else 0,
				i["num_beds"] if i["num_beds"] is not None else 0,
				i["num_bedrooms"] if i["num_bedrooms"] is not None else 0,
				i["accommodates"] if i["accommodates"] is not None else 0,
				i["price"] if i["price"] is not None else 0
				]),				
				i["y"]) for i in data]

			# Split the data up into "batches"
			data = more_itertools.chunked(data, self.batch_size)

			# Format our batches in the way Keras expects them:
			# An array of tuples (x_batch, y_batch)

			# An x_batch is a numpy array of shape (batch_size, 12), 
			# containing the concatenated style and style_preference vectors. 

			# A y_batch is a numpy array of shape (batch_size,1) containing the review scores.

			self.data = [ (np.array([j[0] for j in i]), np.array([j[1] for j in i])) for i in data]
			print(self.data)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		return self.data[idx]
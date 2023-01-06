from pymongo import MongoClient

cluster = "mongodb://localhost:27017/"
client = MongoClient(cluster)
database = client.swasthya


class TabletDatabase:
    def __init__(self):
        self.db = database
        self.collection = self.db.tablets

    def get_details(self, tablet):
        return self.collection.find_one({"tablet": tablet})

    def add_details(self, values):
        if self.in_database(values["tablet"]):
            self.collection.replace_one({"tablet": values["tablet"]}, values)
        else:
            self.collection.insert_one(values)

    def in_database(self, name):
        tab = self.collection.find_one({"tablet": name})
        if tab is None:
            return False
        return True

    def get_first(self):
        return self.collection.find_one()

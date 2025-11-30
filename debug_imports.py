try:
    print("Importing server...")
    from retrofitkit.api import server
    print("Success!")
except Exception as e:
    print("Error importing server: {}".format(e))

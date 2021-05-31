import urllib.request

def get_project_id():
  url = "http://metadata.google.internal/computeMetadata/v1/project/project-id"
  req = urllib.request.Request(url)
  req.add_header("Metadata-Flavor", "Google")
  project_id = urllib.request.urlopen(req).read().decode()
  return project_id

"""
Authentication method to the National Snow and Ice Data Centre (NSIDC)'s
Open-source Project for a Network Data Access Protocol (OPeNDAP) service.

Based on code snippet in https://github.com/pydap/pydap/issues/188
"""

import os
import urllib

import requests
from requests.packages.urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

sessions = {}


class URSSession(requests.Session):
    def __init__(self, username=None, password=None):
        super(URSSession, self).__init__()
        self.username = username
        self.password = password
        self.original_url = None

    def authenticate(self, url):
        self.original_url = url
        super(URSSession, self).get(url)
        self.original_url = None

    def get_redirect_target(self, resp):
        if resp.is_redirect:
            if resp.headers["location"] == self.original_url:
                # Redirected back to original URL, so OAuth2 complete. Exit here
                return None
        return super(URSSession, self).get_redirect_target(resp)

    def rebuild_auth(self, prepared_request, response):
        # If being redirected to URS and we have credentials, add them in
        # otherwise default session code will look to pull from .netrc
        if (
            "https://urs.earthdata.nasa.gov" in prepared_request.url
            and self.username
            and self.password
        ):
            prepared_request.prepare_auth((self.username, self.password))
        else:
            super(URSSession, self).rebuild_auth(prepared_request, response)
        return


def get_session(url):
    """ Get existing session for host or create it
    """
    global sessions
    host = urllib.parse.urlsplit(url).netloc

    if host not in sessions:
        session = requests.Session()
        if "urs" in session.get(url).url:
            session = URSSession(
                os.environ["EARTHDATA_USER"], os.environ["EARTHDATA_PASS"]
            )
            session.authenticate(url)

        retries = Retry(
            total=5,
            connect=3,
            backoff_factor=1,
            method_whitelist=False,
            status_forcelist=[400, 401, 403, 404, 408, 500, 502, 503, 504],
        )
        session.mount("http", HTTPAdapter(max_retries=retries))

        sessions[host] = session

    return sessions[host]


def test_nsidc_opendap():
    """
    """
    import base64
    import xarray as xr

    os.environ["EARTHDATA_USER"] = "weiji14"
    os.environ["EARTHDATA_PASS"] = str(base64.b64decode(s="RWQyNzkxOCMj"))

    url = "https://n5eil02u.ecs.nsidc.org/opendap/ATLAS/ATL06.003/2020.04.04/ATL06_20200404002759_01290710_003_01.h5"
    session = get_session(url=url)

    store = xr.backends.PydapDataStore.open(url=url, session=session)

    xr.open_dataset(store)

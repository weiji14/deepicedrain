#!/usr/bin/env python
# ----------------------------------------------------------------------------
# NSIDC Data Download Script
#
# Copyright (c) 2021 Regents of the University of Colorado
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# Tested in Python 2.7 and Python 3.4, 3.6, 3.7
#
# To run the script at a Linux, macOS, or Cygwin command-line terminal:
#   $ python nsidc-download-ATL11.py
#
# On Windows, open Start menu -> Run and type cmd. Then type:
#     python nsidc-download-ATL11.py
#
# The script will first search Earthdata for all matching files.
# You will then be prompted for your Earthdata username/password
# and the script will download the matching files.
#
# If you wish, you may store your Earthdata username/password in a .netrc
# file in your $HOME directory and the script will automatically attempt to
# read this file. The .netrc file should have the following format:
#    machine urs.earthdata.nasa.gov login myusername password mypassword
# where 'myusername' and 'mypassword' are your Earthdata credentials.
#
from __future__ import print_function

import base64
import itertools
import json
import netrc
import ssl
import sys
from getpass import getpass

try:
    from urllib.parse import urlparse
    from urllib.request import urlopen, Request, build_opener, HTTPCookieProcessor
    from urllib.error import HTTPError, URLError
except ImportError:
    from urlparse import urlparse
    from urllib2 import (
        urlopen,
        Request,
        HTTPError,
        URLError,
        build_opener,
        HTTPCookieProcessor,
    )

short_name = "ATL11"
version = "003"
time_start = "2019-03-29T00:00:00Z"
time_end = "2021-04-21T20:50:43Z"
bounding_box = ""
polygon = ""
filename_filter = "*1?_0?0?_003_0?.h5"
url_list = []

CMR_URL = "https://cmr.earthdata.nasa.gov"
URS_URL = "https://urs.earthdata.nasa.gov"
CMR_PAGE_SIZE = 2000
CMR_FILE_URL = (
    "{0}/search/granules.json?provider=NSIDC_ECS"
    "&sort_key[]=start_date&sort_key[]=producer_granule_id"
    "&scroll=true&page_size={1}".format(CMR_URL, CMR_PAGE_SIZE)
)


def get_username():
    username = ""

    # For Python 2/3 compatibility:
    try:
        do_input = raw_input  # noqa
    except NameError:
        do_input = input

    while not username:
        try:
            username = do_input("Earthdata username: ")
        except KeyboardInterrupt:
            quit()
    return username


def get_password():
    password = ""
    while not password:
        try:
            password = getpass("password: ")
        except KeyboardInterrupt:
            quit()
    return password


def get_credentials(url):
    """Get user credentials from .netrc or prompt for input."""
    credentials = None
    errprefix = ""
    try:
        info = netrc.netrc()
        username, account, password = info.authenticators(urlparse(URS_URL).hostname)
        errprefix = "netrc error: "
    except Exception as e:
        if "No such file" not in str(e):
            print("netrc error: {0}".format(str(e)))
        username = None
        password = None

    while not credentials:
        if not username:
            username = get_username()
            password = get_password()
        credentials = "{0}:{1}".format(username, password)
        credentials = base64.b64encode(credentials.encode("ascii")).decode("ascii")

        if url:
            try:
                req = Request(url)
                req.add_header("Authorization", "Basic {0}".format(credentials))
                opener = build_opener(HTTPCookieProcessor())
                opener.open(req)
            except HTTPError:
                print(errprefix + "Incorrect username or password")
                errprefix = ""
                credentials = None
                username = None
                password = None

    return credentials


def build_version_query_params(version):
    desired_pad_length = 3
    if len(version) > desired_pad_length:
        print('Version string too long: "{0}"'.format(version))
        quit()

    version = str(int(version))  # Strip off any leading zeros
    query_params = ""

    while len(version) <= desired_pad_length:
        padded_version = version.zfill(desired_pad_length)
        query_params += "&version={0}".format(padded_version)
        desired_pad_length -= 1
    return query_params


def filter_add_wildcards(filter):
    if not filter.startswith("*"):
        filter = "*" + filter
    if not filter.endswith("*"):
        filter = filter + "*"
    return filter


def build_filename_filter(filename_filter):
    filters = filename_filter.split(",")
    result = "&options[producer_granule_id][pattern]=true"
    for filter in filters:
        result += "&producer_granule_id[]=" + filter_add_wildcards(filter)
    return result


def build_cmr_query_url(
    short_name,
    version,
    time_start,
    time_end,
    bounding_box=None,
    polygon=None,
    filename_filter=None,
):
    params = "&short_name={0}".format(short_name)
    params += build_version_query_params(version)
    params += "&temporal[]={0},{1}".format(time_start, time_end)
    if polygon:
        params += "&polygon={0}".format(polygon)
    elif bounding_box:
        params += "&bounding_box={0}".format(bounding_box)
    if filename_filter:
        params += build_filename_filter(filename_filter)
    return CMR_FILE_URL + params


def cmr_download(urls):
    """Download files from list of urls."""
    if not urls:
        return

    url_count = len(urls)
    print("Downloading {0} files...".format(url_count))
    credentials = None

    for index, url in enumerate(urls, start=1):
        if not credentials and urlparse(url).scheme == "https":
            credentials = get_credentials(url)

        filename = url.split("/")[-1]
        print(
            "{0}/{1}: {2}".format(
                str(index).zfill(len(str(url_count))), url_count, filename
            )
        )

        try:
            # In Python 3 we could eliminate the opener and just do 2 lines:
            # resp = requests.get(url, auth=(username, password))
            # open(filename, 'wb').write(resp.content)
            req = Request(url)
            if credentials:
                req.add_header("Authorization", "Basic {0}".format(credentials))
            opener = build_opener(HTTPCookieProcessor())
            data = opener.open(req).read()
            open(filename, "wb").write(data)
        except HTTPError as e:
            print("HTTP error {0}, {1}".format(e.code, e.reason))
        except URLError as e:
            print("URL error: {0}".format(e.reason))
        except IOError:
            raise
        except KeyboardInterrupt:
            quit()


def cmr_filter_urls(search_results):
    """Select only the desired data files from CMR response."""
    if "feed" not in search_results or "entry" not in search_results["feed"]:
        return []

    entries = [e["links"] for e in search_results["feed"]["entry"] if "links" in e]
    # Flatten "entries" to a simple list of links
    links = list(itertools.chain(*entries))

    urls = []
    unique_filenames = set()
    for link in links:
        if "href" not in link:
            # Exclude links with nothing to download
            continue
        if "inherited" in link and link["inherited"] is True:
            # Why are we excluding these links?
            continue
        if "rel" in link and "data#" not in link["rel"]:
            # Exclude links which are not classified by CMR as "data" or "metadata"
            continue

        if "title" in link and "opendap" in link["title"].lower():
            # Exclude OPeNDAP links--they are responsible for many duplicates
            # This is a hack; when the metadata is updated to properly identify
            # non-datapool links, we should be able to do this in a non-hack way
            continue

        filename = link["href"].split("/")[-1]
        if filename in unique_filenames:
            # Exclude links with duplicate filenames (they would overwrite)
            continue
        unique_filenames.add(filename)

        urls.append(link["href"])

    return urls


def cmr_search(
    short_name,
    version,
    time_start,
    time_end,
    bounding_box="",
    polygon="",
    filename_filter="",
):
    """Perform a scrolling CMR query for files matching input criteria."""
    cmr_query_url = build_cmr_query_url(
        short_name=short_name,
        version=version,
        time_start=time_start,
        time_end=time_end,
        bounding_box=bounding_box,
        polygon=polygon,
        filename_filter=filename_filter,
    )
    print("Querying for data:\n\t{0}\n".format(cmr_query_url))

    cmr_scroll_id = None
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    try:
        urls = []
        while True:
            req = Request(cmr_query_url)
            if cmr_scroll_id:
                req.add_header("cmr-scroll-id", cmr_scroll_id)
            response = urlopen(req, context=ctx)
            if not cmr_scroll_id:
                # Python 2 and 3 have different case for the http headers
                headers = {k.lower(): v for k, v in dict(response.info()).items()}
                cmr_scroll_id = headers["cmr-scroll-id"]
                hits = int(headers["cmr-hits"])
                if hits > 0:
                    print("Found {0} matches.".format(hits))
                else:
                    print("Found no matches.")
            search_page = response.read()
            search_page = json.loads(search_page.decode("utf-8"))
            url_scroll_results = cmr_filter_urls(search_page)
            if not url_scroll_results:
                break
            if hits > CMR_PAGE_SIZE:
                print(".", end="")
                sys.stdout.flush()
            urls += url_scroll_results

        if hits > CMR_PAGE_SIZE:
            print()
        return urls
    except KeyboardInterrupt:
        quit()


def main():
    global short_name, version, time_start, time_end, bounding_box, polygon, filename_filter, url_list

    # Supply some default search parameters, just for testing purposes.
    # These are only used if the parameters aren't filled in up above.
    if "short_name" in short_name:
        short_name = "MOD10A2"
        version = "6"
        time_start = "2001-01-01T00:00:00Z"
        time_end = "2019-03-07T22:09:38Z"
        bounding_box = ""
        polygon = "-109,37,-102,37,-102,41,-109,41,-109,37"
        filename_filter = "A2019"
        url_list = []

    if not url_list:
        url_list = cmr_search(
            short_name,
            version,
            time_start,
            time_end,
            bounding_box=bounding_box,
            polygon=polygon,
            filename_filter=filename_filter,
        )

    # cmr_download(url_list)
    return url_list


if __name__ == "__main__":
    url_list = main()
    url_list = [
        f'{url.replace("DP9/", "")}\n' for url in url_list if not url.endswith("xml")
    ]
    with open(file="ATL11_to_download.txt", mode="w") as f:
        f.writelines(url_list)

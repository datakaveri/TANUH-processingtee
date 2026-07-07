from urllib.parse import urlparse

from policy.policy_loader import load_policy


def _hostname(host_or_url):
    if host_or_url.startswith("http://") or host_or_url.startswith("https://"):
        return urlparse(host_or_url).hostname

    return host_or_url


def _match_rule(host):

    policy = load_policy()

    for rule in policy["rules"]:

        destinations = rule.get("destinations")

        if destinations is None:
            destinations = [rule.get("destination")]

        if host in destinations:
            return rule

    return None


def is_download_allowed(host_or_url):

    host = _hostname(host_or_url)

    rule = _match_rule(host)

    if rule is None:
        return False

    if "download" in rule:
        return rule["download"].upper() == "ALLOW"

    return rule.get("action", "").upper() == "ALLOW"



def is_upload_allowed(host_or_url):

    host = _hostname(host_or_url)

    rule = _match_rule(host)

    if rule is None:
        return False

    if "upload" in rule:
        return rule["upload"].upper() == "ALLOW"

    return rule.get("action", "").upper() == "ALLOW"
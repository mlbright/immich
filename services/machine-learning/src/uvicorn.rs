use std::net::{IpAddr, Ipv4Addr, SocketAddr};

use anyhow::{Result, anyhow, bail};
use axum::http::Request;
use ipnet::IpNet;
use percent_encoding::percent_decode_str;

pub struct TrustedHosts {
    always_trust: bool,
    networks: Vec<IpNet>,
}

impl TrustedHosts {
    pub fn parse(forwarded_allow_ips: &str) -> Result<Self> {
        let hosts: Vec<&str> = forwarded_allow_ips
            .trim()
            .split(',')
            .filter(|host| !host.is_empty())
            .map(str::trim)
            .collect();
        let networks = hosts
            .iter()
            .filter(|host| **host != "*")
            .map(|host| ip_network(host))
            .collect::<Result<_>>()?;
        Ok(Self {
            always_trust: hosts == ["*"],
            networks,
        })
    }

    fn contains(&self, host: IpAddr) -> bool {
        self.always_trust || self.networks.iter().any(|network| network.contains(&host))
    }
}

fn ip_network(address: &str) -> Result<IpNet> {
    let invalid = || anyhow!("'{address}' does not appear to be an IPv4 or IPv6 network");
    let (host, mask) = match address.split_once('/') {
        Some((host, mask)) => (host, Some(mask)),
        None => (address, None),
    };
    let host: IpAddr = host.parse().map_err(|_| invalid())?;
    let max_prefix_len = if host.is_ipv4() { 32 } else { 128 };
    let prefix_len = match mask {
        Some(mask) => prefix_len(host, mask, max_prefix_len).ok_or_else(invalid)?,
        None => max_prefix_len,
    };
    let network = IpNet::new(host, prefix_len).map_err(|_| invalid())?;
    if network.trunc() != network {
        bail!("{host}/{prefix_len} has host bits set");
    }
    Ok(network)
}

fn prefix_len(host: IpAddr, mask: &str, max_prefix_len: u8) -> Option<u8> {
    if !mask.is_empty() && mask.bytes().all(|byte| byte.is_ascii_digit()) {
        return mask
            .parse::<u8>()
            .ok()
            .filter(|prefix_len| *prefix_len <= max_prefix_len);
    }
    if host.is_ipv6() {
        return None;
    }
    let netmask = u32::from(mask.parse::<Ipv4Addr>().ok()?);
    [netmask, !netmask]
        .into_iter()
        .find(|bits| bits.leading_ones() + bits.trailing_zeros() == 32)
        .map(|bits| bits.leading_ones() as u8)
}

pub fn latin1(bytes: &[u8]) -> String {
    bytes.iter().map(|&byte| char::from(byte)).collect()
}

pub struct Scope {
    pub scheme: String,
    pub server: SocketAddr,
    pub path: String,
}

impl Scope {
    pub fn new<B>(
        request: &Request<B>,
        server: SocketAddr,
        client: SocketAddr,
        trusted_hosts: &TrustedHosts,
    ) -> Self {
        let raw_path = request.uri().path();
        let path = if raw_path.contains('%') {
            percent_decode_str(raw_path)
                .decode_utf8_lossy()
                .into_owned()
        } else {
            raw_path.to_owned()
        };
        Self {
            scheme: forwarded_proto(request, client, trusted_hosts)
                .unwrap_or_else(|| "http".to_owned()),
            server,
            path,
        }
    }
}

fn forwarded_proto<B>(
    request: &Request<B>,
    client: SocketAddr,
    trusted_hosts: &TrustedHosts,
) -> Option<String> {
    if !trusted_hosts.contains(client.ip()) {
        return None;
    }
    let value = request
        .headers()
        .get_all("x-forwarded-proto")
        .iter()
        .next_back()?;
    let proto = latin1(value.as_bytes()).trim().to_owned();
    matches!(proto.as_str(), "http" | "https" | "ws" | "wss").then_some(proto)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn error(value: &str) -> String {
        TrustedHosts::parse(value)
            .err()
            .map(|error| error.to_string())
            .unwrap_or_default()
    }

    fn trusts(value: &str, host: &str) -> bool {
        TrustedHosts::parse(value)
            .unwrap_or_else(|error| panic!("{value}: {error}"))
            .contains(
                host.parse()
                    .unwrap_or_else(|error| panic!("{host}: {error}")),
            )
    }

    #[test]
    fn forwarded_allow_ips_are_validated_as_gunicorn_does() {
        assert_eq!(
            error("bogus"),
            "'bogus' does not appear to be an IPv4 or IPv6 network"
        );
        assert_eq!(error("10.0.0.1/8"), "10.0.0.1/8 has host bits set");
        assert_eq!(
            error("10.0.0.0/8, ,::1"),
            "'' does not appear to be an IPv4 or IPv6 network"
        );
        assert_eq!(
            error("::/0.0.0.0"),
            "'::/0.0.0.0' does not appear to be an IPv4 or IPv6 network"
        );
        assert_eq!(
            error("10.0.0.0/33"),
            "'10.0.0.0/33' does not appear to be an IPv4 or IPv6 network"
        );
        assert_eq!(error(""), "");
        assert_eq!(error(" 127.0.0.1,,::1, "), "");
    }

    #[test]
    fn trusted_hosts_match_addresses_and_networks() {
        assert!(trusts("127.0.0.1,::1", "127.0.0.1"));
        assert!(trusts("127.0.0.1,::1", "::1"));
        assert!(!trusts("127.0.0.1,::1", "::ffff:127.0.0.1"));
        assert!(!trusts("127.0.0.1,::1", "10.0.0.1"));
        assert!(trusts("10.0.0.0/8", "10.200.3.4"));
        assert!(trusts("10.0.0.0/255.0.0.0", "10.200.3.4"));
        assert!(trusts("10.0.0.0/0.255.255.255", "10.200.3.4"));
        assert!(!trusts("10.0.0.0/255.0.0.0", "11.0.0.1"));
        assert!(trusts("*", "203.0.113.9"));
        assert!(!trusts("*,127.0.0.1", "203.0.113.9"));
        assert!(!trusts("", "127.0.0.1"));
    }
}

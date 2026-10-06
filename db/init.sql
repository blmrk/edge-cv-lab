-- Grafana's datasource role: reads the lab tables, no table writes (SECURITY.md, Lab-only defaults).
-- Postgres runs this once, as postgres in database lab, only when its data volume is empty; `make down` removes it.
-- ingest creates its tables later, as postgres, so the default privileges below cover them as they appear.
CREATE ROLE grafana_ro LOGIN PASSWORD 'lab';
GRANT CONNECT ON DATABASE lab TO grafana_ro;
GRANT USAGE ON SCHEMA public TO grafana_ro;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO grafana_ro;
ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public GRANT SELECT ON TABLES TO grafana_ro;
REVOKE TEMPORARY ON DATABASE lab FROM PUBLIC;  -- no temp tables either; postgres, a superuser, keeps them
REVOKE CONNECT ON DATABASE postgres FROM PUBLIC;  -- nor logins to the maintenance database; postgres, a superuser, still connects
REVOKE CONNECT ON DATABASE template1 FROM PUBLIC;  -- nor to template1, where the revokes above (per database) do not apply
-- nor large objects, which any role may otherwise create and then write; postgres keeps these too
REVOKE EXECUTE ON FUNCTION lo_create(oid), lo_creat(integer), lo_from_bytea(oid, bytea), lo_put(oid, bigint, bytea),
  lowrite(integer, bytea) FROM PUBLIC;
-- Not revocable in Postgres 16: any role may ALTER ROLE itself, its own password and session defaults.

# Databricks notebook source
# DBTITLE 1,Grant LPG app SP access to secrets
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
from databricks.sdk.service.workspace import AclPermission
w.secrets.put_acl(scope="commodity-news-secrets", principal="34723e4c-6c84-42cb-8014-16478ddd2599", permission=AclPermission.READ)
print("Done — LPG app can now read Azure OpenAI credentials")

# COMMAND ----------



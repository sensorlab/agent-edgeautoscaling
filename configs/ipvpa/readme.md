New restartPolicy for resize, it gives users control over how their containers are handled when resources are resized.
If it's set to NotRequired, patching won't restart the container:

```shell
kubectl patch pod ray-worker-pod-rasp1 --patch '{"spec":{"containers":[{"name":"ray-worker", "resources":{"requests":{"cpu":"500m"}, "limits":{"cpu":"500m"}}}]}}'
```
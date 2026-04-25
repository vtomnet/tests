URL: https://download.fedoraproject.org/pub/fedora/linux/releases/43/Cloud/x86_64/images/Fedora-Cloud-Base-Generic-43-1.6.x86_64.qcow2

Dependencies:
- qemu
- cdrtools on macOS

## Run

```bash
curl -Lo fedora.qcow2 $URL
qemu-img create -f qcow2 -b fedora.qcow2 -B qcow2 temp.qcow2
(cd seed; mkisofs -V cidata -J -r -o ../seed.iso user-data meta-data)
qemu-system-x86_64 \
  -serial mon:stdio \
  -display none \
  -m 4096 \
  -smp 2 \
  -snapshot \
  -drive file=temp.qcow2,if=virtio,format=qcow2 \
  -drive file=seed.iso,media=cdrom,format=raw,readonly=on \
  -nic user,hostfwd=tcp::2222-:22
```

## Clean

```bash
rm seed.iso temp.qcow2
```

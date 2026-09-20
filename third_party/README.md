# RoboDojo third-party checkouts

The RoboDojo simulator and XPolicyLab policy-server/client are intentionally
kept as separate checkouts. Their source trees are ignored by the parent Git
repository because they have their own histories and licenses. Recreate them
at the tested revisions with:

```bash
./scripts/setup_robodojo.sh
```

The revisions and upstream URLs are recorded in `manifest.json`. RoboDojo is
released for non-commercial research, education, and evaluation; read both
upstream licenses before redistributing a checkout.

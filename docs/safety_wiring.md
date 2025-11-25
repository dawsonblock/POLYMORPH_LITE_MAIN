# Safety Wiring – Retrofit Kit

- E-Stop loop → safety relay → PSU enable (normally-closed loop).
- Door interlock → DI input; optional series in E-Stop.
- Watchdog relay: software heartbeat toggles DO; failure drops relay.

This repo ships software checks; certification requires approved hardware + assessment.

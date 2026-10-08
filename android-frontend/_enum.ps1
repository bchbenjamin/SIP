dir 'C:\Shared\Projects\SIP-Android\backend\src\main\java\com\sip\backend' -Directory | % { =; Write-Output ('DIR: '+.Name); dir .FullName -File | % { Write-Output ('  FILE: '+.Name) } }

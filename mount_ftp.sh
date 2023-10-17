tgt="/SOC_DATA/건물균열프로젝트/Gkes/지케스 가공데이터"
#curlftpfs pcndev.co.kr:10200 ./nas_pcn -o user=dinsight:Elqdlstkdlxm2!,allow_other,ftp_port=10200
#curlftpfs dinsight:Elqdlstkdlxm2!@pcndev.co.kr ./nas_pcn
#sshfs -o reconnect -p 10200 dinsight@pcndev.co.kr:/ ./nas_pcn #<<< "Elqdlstkdlxm2!"
sshfs -p 10200 dinsight@pcndev.co.kr:/SOC_DATA/건물균열프로젝트/Gkes/지케스\ 가공데이터 ./nas_pcn #<<< "Elqdlstkdlxm2!"

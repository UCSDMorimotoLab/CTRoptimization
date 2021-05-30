def log(count,multiplier,i,flag,error):
    f= open("collision_log.txt","a+")
    f.write("collision %d\r\n" % (count))
    f.write("multiplier %d\r\n" % (multiplier))
    f.write("i %d\r\n" % (i))
    f.write("flag %d\r\n" % (flag))
    f.write("error %d\r\n" % (error))
    f.close()
    

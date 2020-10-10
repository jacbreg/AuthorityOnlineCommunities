# script to study PCA of individual characteristics of ell-stackexchange users

library(FactoMineR)

dta <- read.csv("/home/jacopo/OneDrive/Dati_Jac_locali/stack/ell/individual_chars_dummies.csv")

 set.seed(111) # set seed for replicability

# # K-MEANS CLUSTERING - NOT really a good idea, because the concept of means in dummies is not well defined. Also, it doesn't work
# too well, with different initial random state the output may be changing a lot
# kl = kmeans(dta[,2:11], centers=3, nstart = 100)
# # check number of obs per group
# table(kl$cluster)
# # vector with groups indicators
# clusters <- kl$cluster
# # add group variable
# dta$r_groups <- clusters


# MCA (Multiple Correspondence Analysis)
# it seems it needs factorial/categorical variables
str(dta) # to check data types
dtaf <- data.frame(dta) # create a duplicate that doesn't point to the original object

# transform in string variable the dummies (excluding user id)
for (i in 2:8) {
  dtaf[,i] <- as.character(dtaf[,i])
}
# transform in factorial variable (excluding user id)
for (i in 2:8) {
  dtaf[,i] <- as.factor(dtaf[,i])
}

# train using only dummies and not clusters' variable
res.mca = MCA(dtaf[,2:8], graph = FALSE)
# plot variables on first two components
plot.MCA(res.mca, label=c("var"), invisible = c("quali.sup"), choix=c("var"))
#plot.MCA(res.mca, label=c("var"), invisible = c("quali.sup"), choix=c("var"), axes = c(2,3))

# plot individuals on first two components, colored by k-means clustering group with variables - R GROUPS
plot.MCA(res.mca, label=c("var"), invisible = c("quali.sup"), choix=c("ind")) 
# EQUIVALENT TO plot(res.mca$ind$coord[,1],res.mca$ind$coord[,2])

# compute clusters use k-means algorithm on MCA components
cl <- kmeans(res.mca$ind$coord, centers=3, nstart = 20)
table(cl$cluster)
dtaf$user_types <- as.factor(cl$cluster)
res.mca = MCA(dtaf[,2:9], graph = FALSE, quali.sup = 8) # just necessary to include groups in sample so to be able to plot different colours
plot.MCA(res.mca, label=c("none"), choix=c("ind"), habillage = 'user_types')
#plot.MCA(res.mca, label=c("none"), choix=c("ind"), habillage = 'user_types', axes = c(2,3))

# save
dta$user_types <- cl$cluster
write.csv(dta, "/home/jacopo/OneDrive/Dati_Jac_locali/stack/ell/individual_chars_dummies_wgroups.csv")